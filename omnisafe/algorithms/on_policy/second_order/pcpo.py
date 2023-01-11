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
"""Implementation of the PCPO algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class PCPO(TRPO):
    """The Projection-Based Constrained Policy Optimization (PCPO) algorithm.

    References:
        Title: Projection-Based Constrained Policy Optimization
        Authors: Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, Peter J. Ramadge.
        URL: https://arxiv.org/abs/2010.03152
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize PCPO.

        PCPO is a derivative of TRPO.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)
        self.cost_limit = self.cfgs.cost_limit

    # pylint: disable-next=too-many-locals,too-many-arguments
    def adjust_cpo_step_direction(
        self,
        step_dir: torch.Tensor,
        g_flat: torch.Tensor,
        cost: torch.Tensor,
        optim_case: int,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
        loss_pi_before: torch.Tensor,
        loss_pi_cost_before: torch.Tensor,
        total_steps: int = 25,
        decay: float = 0.8,
    ) -> Tuple[torch.Tensor, int]:
        r"""Use line-search to find the step size that satisfies the constraint.

        PCPO uses line-search to find the step size that satisfies the constraint.
        The constraint is defined as:

        .. math::
            J^C(\theta + \alpha \delta) - J^C(\theta) \leq \max \{0, c\}\\
            D_{KL}(\pi_{\theta}(\cdot|s) || \pi_{\theta + \alpha \delta}(\cdot|s)) \leq \delta_{KL}

        where :math:`\delta_{KL}` is the constraint of KL divergence, :math:`\alpha` is the step size,
        :math:`c` is the violation of constraint.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            cost_adv (torch.Tensor): The cost advantage.
            loss_pi_before (torch.Tensor): The loss of the policy before the step.
            loss_pi_cost_before (torch.Tensor): The loss of the cost before the step.
            total_steps (int, optional): The total steps of line-search. Defaults to 25.
            decay (float, optional): The decay of step size. Defaults to 0.8.
        """
        step_frac = 1.0
        _theta_old = get_flat_params_from(self.actor_critic.actor)
        expected_rew_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            set_param_values_to_model(self.actor_critic.actor, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                # Loss of policy reward from target/expected reward
                loss_pi_rew, _ = self.compute_loss_pi(obs=obs, act=act, log_p=log_p, adv=adv)
                # Loss of cost of policy cost from real/expected reward
                loss_pi_cost, _ = self.compute_loss_cost_performance(
                    obs=obs, act=act, log_p=log_p, cost_adv=cost_adv
                )
                self.loss_record.append(loss_pi=(loss_pi_rew.mean() + loss_pi_cost.mean()).item())
                # determine KL div between new and old policy
                q_dist = self.actor_critic.actor(obs)
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_rew_improve = loss_pi_before - loss_pi_rew.item()
            cost_diff = loss_pi_cost.item() - loss_pi_cost_before

            # Average across MPI processes...
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            loss_rew_improve = distributed_utils.mpi_avg(loss_rew_improve)
            cost_diff = distributed_utils.mpi_avg(cost_diff)
            menu = (expected_rew_improve, loss_rew_improve)
            self.logger.log(f'Expected Improvement: {menu[0]} Actual: {menu[1]}')

            if not torch.isfinite(loss_pi_rew) and not torch.isfinite(loss_pi_cost):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_rew_improve < 0 if optim_case > 1 else False:
                self.logger.log('INFO: did not improve improve <0')

            elif cost_diff > max(-cost, 0):
                self.logger.log(f'INFO: no improve {cost_diff} > {max(-cost, 0)}')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log(f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self.logger.log(f'Accept step at i={j + 1}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.actor_critic.actor, _theta_old)
        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self) -> None:
        r"""Log the PCPO specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Misc/cost_gradient_norm
                -   The norm of the cost gradient.
            *   -   Misc/q
                -   The :math:`q` vector, which is the conjugate of Hessian :math:`H`.
            *   -   Misc/r
                -   The :math:`r` vector, where :math:`r = g^T H^{-1} b`.
            *   -   Misc/s
                -   The :math:`s` vector, where :math:`s = b^T H^{-1} b`
            *   -   Misc/A
                -   The A matrix, where :math:`A = q - \frac{r^2}{s}`
            *   -   Misc/B
                -   The B matrix, where :math:`B = 2 \delta_{KL} - \frac{c^2}{s}` ,
                    where :math:`c` is the cost violation in current epoch, and
                    :math:`\delta_{KL}` is the target KL divergence.
            *   -   Misc/Lambda_star
                -   The :math:`\lambda^*` vector.
            *   -   Misc/Nu_star
                -   The :math:`\nu^*` vector.
            *   -   Misc/OptimCase
                -   The optimization case.
        """
        TRPO.algorithm_specific_logs(self)
        self.logger.log_tabular('Misc/cost_gradient_norm')
        self.logger.log_tabular('Misc/A')
        self.logger.log_tabular('Misc/B')
        self.logger.log_tabular('Misc/q')
        self.logger.log_tabular('Misc/r')
        self.logger.log_tabular('Misc/s')
        self.logger.log_tabular('Misc/Lambda_star')
        self.logger.log_tabular('Misc/Nu_star')
        self.logger.log_tabular('Misc/OptimCase')

    def compute_loss_cost_performance(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the loss of cost of policy cost from real cost.

        .. math::
            L = \mathbb{E}_{\pi} \left[ \frac{\pi(a|s)}{\pi_{old}(a|s)} A^C(s, a) \right]

        where :math:`A^C(s, a)` is the cost advantage,
        :math:`\pi_{old}(a|s)` is the old policy,
        :math:`\pi(a|s)` is the current policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            cost_adv (torch.Tensor): Cost advantage.
        """
        _, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)
        cost_loss = (ratio * cost_adv).mean()
        info = {}
        return cost_loss, info

    # pylint: disable=too-many-locals,invalid-name,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        """Update policy network.

        PCPO updates policy network using the conjugate gradient algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint. (Both KL divergence and cost limit).
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            cost_adv (torch.Tensor): The cost advantage tensor.
        """
        self.fvp_obs = obs[::4]
        theta_old = get_flat_params_from(self.actor_critic.actor)
        self.actor_optimizer.zero_grad()
        # Process the advantage function.
        processed_adv = self.compute_surrogate(adv=adv, cost_adv=cost_adv)
        # Compute the loss of policy net.
        loss_pi, pi_info = self.compute_loss_pi(obs=obs, act=act, log_p=log_p, adv=processed_adv)
        loss_pi_before = loss_pi.item()
        # get prob. distribution before updates
        p_dist = self.actor_critic.actor(obs)
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor)
        g_flat = get_flat_gradients_from(self.actor_critic.actor)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        H_inv_g = self.Fvp(x)
        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))
        assert xHx.item() >= 0, 'No negative values'

        # get the policy cost performance gradient b (flat as vector)
        self.actor_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(
            obs=obs, act=act, log_p=log_p, cost_adv=cost_adv
        )
        loss_cost.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor)
        loss_pi_cost_before = loss_cost.item()
        b_flat = get_flat_gradients_from(self.actor_critic.actor)

        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        cost = ep_costs - self.cost_limit
        cost /= self.logger.get_stats('Metrics/EpLen')[0] + eps  # rescale
        self.logger.log(f'c = {cost}')
        self.logger.log(f'b^T b = {b_flat.dot(b_flat).item()}')

        # set variable names as used in the paper
        p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx
        # g^T H^{-1} b
        r = g_flat.dot(p)
        # b^T H^{-1} b
        s = b_flat.dot(p)
        step_dir = (
            torch.sqrt(2 * self.target_kl / (q + 1e-8)) * H_inv_g
            - torch.clamp_min(
                (torch.sqrt(2 * self.target_kl / q) * r + cost) / s, torch.tensor(0.0)
            )
            * p
        )

        final_step_dir, accept_step = self.adjust_cpo_step_direction(
            step_dir,
            g_flat,
            cost=cost,
            optim_case=2,
            p_dist=p_dist,
            obs=obs,
            act=act,
            log_p=log_p,
            adv=adv,
            cost_adv=cost_adv,
            loss_pi_before=loss_pi_before,
            loss_pi_cost_before=loss_pi_cost_before,
            total_steps=20,
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor, new_theta)

        self.logger.store(
            **{
                'Train/Entropy': pi_info['ent'],
                'Train/PolicyRatio': pi_info['ratio'],
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': final_step_dir.norm().numpy(),
                'Misc/xHx': xHx.numpy(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/cost_gradient_norm': torch.norm(b_flat).numpy(),
                'Misc/Lambda_star': 1.0,
                'Misc/Nu_star': 1.0,
                'Misc/OptimCase': int(1),
                'Misc/A': 1.0,
                'Misc/B': 1.0,
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            }
        )
