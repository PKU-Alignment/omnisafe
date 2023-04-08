# Copyright 2023 OmniSafe Team. All Rights Reserved.
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

from __future__ import annotations

import torch
from torch.distributions import Distribution

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
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

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/AcceptanceStep')

    # pylint: disable-next=too-many-arguments,too-many-locals,arguments-differ
    def _search_step_size(
        self,
        step_direction: torch.Tensor,
        grad: torch.Tensor,
        p_dist: Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        loss_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
    ) -> tuple[torch.Tensor, int]:
        """TRPO performs `line-search <https://en.wikipedia.org/wiki/Line_search>`_ until constraint satisfaction.

        .. hint::

            TRPO search around for a satisfied step of policy update to improve loss and reward performance.
            The search is done by line-search, which is a way to find a step size that satisfies the constraint.
            The constraint is the KL-divergence between the old policy and the new policy.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            cost_adv (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grad.dot(step_direction)

        final_kl = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(total_steps):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(self._actor_critic.actor, new_theta)

            with torch.no_grad():
                loss, _ = self._loss_pi(obs, act, logp, adv)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
                kl = distributed.dist_avg(kl)
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss.item()
            # average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            loss_improve = distributed.dist_avg(loss_improve)
            self._logger.log(f'Expected Improvement: {expected_improve} Actual: {loss_improve}')
            if not torch.isfinite(loss):
                self._logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self._logger.log('INFO: did not improve improve <0')
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                self._logger.log(f'Accept step at i={acceptance_step}')
                final_kl = kl
                break
            step_frac *= decay
        else:
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        set_param_values_to_model(self._actor_critic.actor, theta_old)

        self._logger.store(
            **{
                'Train/KL': final_kl,
            },
        )

        return step_frac * step_direction, acceptance_step

    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
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
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss, info = self._loss_pi(obs, act, logp, adv)
        loss_before = distributed.dist_avg(loss).item()
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grad = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grad, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grad=grad,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv=adv,
            loss_before=loss_before,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss, info = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss.mean().item(),
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grad).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
            },
        )
