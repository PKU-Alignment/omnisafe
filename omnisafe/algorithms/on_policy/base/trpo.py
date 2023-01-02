# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the TRPO algorithm."""

from typing import NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class TRPO(NaturalPG):
    """The Trust Region Policy Optimization (TRPO) algorithm.

    References:
        - Title: Trust Region Policy Optimization
        - Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel.
        - URL: `TRPO <https://arxiv.org/abs/1502.05477>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize Trust Region Policy Optimization.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)

    # pylint: disable-next=too-many-arguments,too-many-locals,arguments-differ
    def search_step_size(
        self,
        step_dir: torch.Tensor,
        g_flat: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
        loss_pi_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
    ) -> Tuple[torch.Tensor, int]:
        """TRPO performs `line-search <https://en.wikipedia.org/wiki/Line_search>`_ until constraint satisfaction.

        .. note::

            TRPO search around for a satisfied step of policy update to improve loss and reward performance.
            The search is done by line-search, which is a way to find a step size that satisfies the constraint.
            The constraint is the KL-divergence between the old policy and the new policy.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            cost_adv (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        _theta_old = get_flat_params_from(self.actor_critic.actor.net)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = g_flat.dot(step_dir)

        # While not within_trust_region and not out of total_steps:
        for j in range(total_steps):
            # Update theta params
            new_theta = _theta_old + step_frac * step_dir
            # Set new params as params of net
            set_param_values_to_model(self.actor_critic.actor.net, new_theta)
            # The stepNo this update accept
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, _ = self.compute_loss_pi(
                    obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
                )
                # Compute KL distance between new and old policy
                q_dist = self.actor_critic.actor(obs)
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            # Real loss improve: old policy loss - new policy loss
            loss_improve = loss_pi_before - loss_pi.item()
            # Average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            loss_improve = distributed_utils.mpi_avg(loss_improve)
            menu = (expected_improve, loss_improve)
            self.logger.log(f'Expected Improvement: {menu[0]} Actual: {menu[1]}')
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.actor_critic.actor.net, _theta_old)

        return step_frac * step_dir, acceptance_step

    # pylint: disable-next=too-many-locals,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        """Update policy network.

        Trust Policy Region Optimization updates policy network using the
        `conjugate gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint.
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            cost_adv (torch.Tensor): The cost advantage tensor.
        """
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.actor_critic.actor.net)
        self.actor_critic.actor.net.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(
            obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
        )
        loss_pi_before = distributed_utils.mpi_avg(loss_pi.item())
        p_dist = self.actor_critic.actor(obs)
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
        g_flat = get_flat_gradients_from(self.actor_critic.actor.net)
        g_flat *= -1

        # pylint: disable-next=invalid-name
        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.search_step_size(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            loss_pi_before=loss_pi_before,
            obs=obs,
            act=act,
            log_p=log_p,
            adv=adv,
            cost_adv=cost_adv,
        )

        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor.net, new_theta)

        with torch.no_grad():
            q_dist = self.actor_critic.actor(obs)
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(
                obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
            )

        self.logger.store(
            **{
                'Values/Adv': adv.numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': kl,
                'Train/PolicyRatio': pi_info['ratio'],
                'Train/StopIter': 1,
                'Loss/Loss_pi': loss_pi.item(),
                'Loss/Delta_loss_pi': loss_pi.item() - loss_pi_before,
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(final_step_dir).numpy(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )
