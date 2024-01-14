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
"""Implementation of the PCPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class PCPO(CPO):
    """The Projection-Based Constrained Policy Optimization (PCPO) algorithm.

    References:
        - Title: Projection-Based Constrained Policy Optimization
        - Authors: Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, Peter J. Ramadge.
        - URL: `PCPO <https://arxiv.org/abs/2010.03152>`_
    """

    # pylint: disable-next=too-many-locals,too-many-arguments
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

        PCPO updates policy network using the conjugate gradient algorithm, following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint. (Both KL divergence and cost limit).
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        # pylint: disable=invalid-name
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        H_inv_g = self._fvp(x)
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        self._actor_critic.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grads = get_flat_gradients_from(self._actor_critic.actor)
        ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

        self._logger.log(f'c = {ep_costs}')
        self._logger.log(f'b^T b = {b_grads.dot(b_grads).item()}')

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        step_direction = (
            torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (q + 1e-8)) * H_inv_g
            - torch.clamp_min(
                (torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / q) * r + ep_costs) / s,
                torch.tensor(0.0, device=self._device),
            )
            * p
        )  # pylint: disable=invalid-name

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=200,
            violation_c=ep_costs,
        )
        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': 1.0,
                'Misc/Nu_star': 1.0,
                'Misc/OptimCase': 1,
                'Misc/A': 1.0,
                'Misc/B': 1.0,
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            },
        )
