# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the P3O algorithm."""

import torch
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed


@registry.register
class P3O(PPO):
    """The Implementation of the P3O algorithm.

    References:
        - Title: Penalized Proximal Policy Optimization for Safe Reinforcement Learning
        - Authors: Linrui Zhang, Li Shen, Long Yang, Shixiang Chen, Bo Yuan, Xueqian Wang, Dacheng Tao.
        - URL: `P3O <https://arxiv.org/pdf/2205.11814.pdf>`_
    """

    def _init_log(self) -> None:
        r"""Log the P3O specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -  ``Loss/Loss_pi_cost``
                -   The loss of the cost performance.
        """
        super()._init_log()
        self._logger.register_key('Loss/Loss_pi_cost', delta=True)

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the loss of cost of policy cost from real cost.

        .. math::
            L = \mathbb{E}_{\pi} \left[ \frac{\pi^{'}(a|s)}{\pi(a|s)} A^{C}_{\pi_\theta}(s, a) \right]

        where :math:`A^{C}_{\pi_\theta}(s, a)` is the cost advantage,
        :math:`\pi(a|s)` is the old policy,
        :math:`\pi^{'}(a|s)` is the current policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            logp (torch.Tensor): Log probability of action.
            adv_c (torch.Tensor): Cost advantage.

        Returns:
            torch.Tensor: The loss of cost of policy cost from real cost.
        """
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        surr_cadv = (ratio * adv_c).mean()
        Jc = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        loss_cost = self._cfgs.algo_cfgs.kappa * F.relu(surr_cadv + Jc)
        return loss_cost.mean()

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        r"""Update policy network under a double for loop.

            The pseudo code is shown below:

            .. code-block:: python

                for _ in range(self.cfgs.actor_iters):
                    for _ in range(self.cfgs.num_mini_batches):
                        # Get mini-batch data
                        # Compute loss
                        # Update network

            .. warning::
                For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
                the ``KL divergence`` between the old policy and the new policy is calculated.
                And the ``KL divergence`` is used to determine whether the update is successful.
                If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log_p`` stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
            adv_c (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        loss_reward, info = self._loss_pi(obs, act, logp, adv_r)
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)

        loss = loss_reward + loss_cost

        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss_reward.mean().item(),
                'Loss/Loss_pi_cost': loss_cost.mean().item(),
            },
        )
