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
from omnisafe.utils import distributed_utils


@registry.register
class P3O(PPO):
    """The Implementation of the IPO algorithm.

    References:
        - Title: Penalized Proximal Policy Optimization for Safe Reinforcement Learning
        - Authors: Linrui Zhang, Li Shen, Long Yang, Shixiang Chen, Bo Yuan, Xueqian Wang, Dacheng Tao.
        - URL: `P3O <https://arxiv.org/pdf/2205.11814.pdf>`_
    """

    def compute_loss_cost_performance(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the loss of the cost performance.

        The loss is defined as:

        .. math::

            \mathcal{L}_{\pi_c} = \kappa \cdot \max
            \left(0, \frac{\pi_c(a_t|s_t)}{\pi(a_t|s_t)} \cdot A_{c_t} + J_c - \bar{J}_c\right)

        where :math:`\kappa` is the penalty coefficient, :math:`\pi_c` is the cost performance,
        :math:`\pi` is the policy, :math:`A_{c_t}` is the cost advantage, :math:`J_c` is the cost
        of the current episode, and :math:`\bar{J}_c` is the cost limit.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            log_p (torch.Tensor): The log probability of the action.
            cost_adv (torch.Tensor): The cost advantage.
        """
        _, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)
        surr_cadv = (ratio * cost_adv).mean()
        Jc = self.logger.get_stats('Metrics/EpCost')[0] - self.cfgs.cost_limit
        loss_pi_c = self.cfgs.kappa * F.relu(surr_cadv + Jc)
        return loss_pi_c.mean()

    # pylint: disable-next=too-many-locals,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
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
            cost_adv (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        # process the advantage function.
        processed_adv = self.compute_surrogate(adv=adv, cost_adv=cost_adv)
        # compute the loss of policy net.
        loss_pi, pi_info = self.compute_loss_pi(obs=obs, act=act, log_p=log_p, adv=processed_adv)
        # compute the cost performance of policy net.
        loss_pi_c = self.compute_loss_cost_performance(
            obs=obs, act=act, log_p=log_p, cost_adv=cost_adv
        )
        # log the loss of policy net.
        self.loss_record.append(loss_pi=(loss_pi - loss_pi_c).mean().item())
        # update the policy net.
        self.actor_optimizer.zero_grad()
        # backward the loss of policy net.
        (loss_pi - loss_pi_c).backward()
        # clip the gradient of policy net.
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
            )
        # average the gradient of policy net.
        distributed_utils.mpi_avg_grads(self.actor_critic.actor)
        self.actor_optimizer.step()
        self.logger.store(
            **{
                'Train/Entropy': pi_info['ent'],
                'Train/PolicyRatio': pi_info['ratio'],
            }
        )
