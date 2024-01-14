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
"""Implementation of the P3O algorithm."""

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

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
        """Log the P3O specific information.

        +-------------------+-----------------------------------+
        | Things to log     | Description                       |
        +===================+===================================+
        | Loss/Loss_pi_cost | The loss of the cost performance. |
        +-------------------+-----------------------------------+
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

        We compute the loss of cost of policy cost from real cost.

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}}  \left[
                \frac{\pi^{'} (a|s)}{\pi (a|s)} A^{C}_{\pi_{\theta}} (s, a)
            \right]

        where :math:`A^{C}_{\pi_{\theta}} (s, a)` is the cost advantage, :math:`\pi (a|s)` is the
        old policy, and :math:`\pi^{'} (a|s)` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The loss of the cost performance.
        """
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        surr_cadv = (ratio * adv_c).mean()
        Jc = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        loss_cost = self._cfgs.algo_cfgs.kappa * F.relu(surr_cadv + Jc)
        self._logger.store({'Loss/Loss_pi_cost': loss_cost.mean().item()})
        return loss_cost.mean()

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        The pseudo code is shown below:

        .. code-block:: python

            for _ in range(self.cfgs.actor_iters):
                for _ in range(self.cfgs.num_mini_batches):
                    # Get mini-batch data
                    # Compute loss
                    # Update network

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.), the ``KL divergence``
            between the old policy and the new policy is calculated. And the ``KL divergence`` is
            used to determine whether the update is successful. If the ``KL divergence`` is too
            large, the update will be terminated.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            logp (torch.Tensor): ``log_p`` stored in buffer.
            adv_r (torch.Tensor): ``reward_advantage`` stored in buffer.
            adv_c (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)

        loss = loss_reward + loss_cost

        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
