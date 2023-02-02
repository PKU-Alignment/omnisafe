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
"""Implementation of the CUP algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.record_queue import RecordQueue
from omnisafe.utils import distributed_utils


@registry.register
class CUP(PPO, Lagrange):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
                 Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize CUP.

        CUP is a combination of :class:`PPO` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        PPO.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        Lagrange.__init__(
            self,
            cost_limit=self.cfgs.lagrange_cfgs.cost_limit,
            lagrangian_multiplier_init=self.cfgs.lagrange_cfgs.lagrangian_multiplier_init,
            lambda_lr=self.cfgs.lagrange_cfgs.lambda_lr,
            lambda_optimizer=self.cfgs.lagrange_cfgs.lambda_optimizer,
            lagrangian_upper_bound=self.cfgs.lagrange_cfgs.lagrangian_upper_bound,
        )
        self.lam = self.cfgs.lam
        self.eta = self.cfgs.eta
        self.max_ratio = 0
        self.min_ratio = 0
        self.p_dist = None
        self.loss_record = RecordQueue('loss_pi', 'loss_v', 'loss_c', 'loss_pi_c', maxlen=100)

    def algorithm_specific_logs(self) -> None:
        """Log the CUP specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
            *   -   Train/MaxRatio
                -   The maximum ratio between the current policy and the old policy.
            *   -   Train/MinRatio
                -   The minimum ratio between the current policy and the old policy.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
        self.logger.log_tabular('Train/MaxRatio', self.max_ratio)
        self.logger.log_tabular('Train/MinRatio', self.min_ratio)
        self.logger.log_tabular('Loss/Loss_pi_c')
        self.logger.log_tabular('Loss/Delta_loss_pi_c')
        self.logger.log_tabular('Train/SecondStepStopIter')
        self.logger.log_tabular('Train/SecondStepEntropy')
        self.logger.log_tabular('Train/SecondStepPolicyRatio')

    # pylint: disable-next=too-many-locals
    def compute_loss_cost_performance(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the KL divergence between the current policy and the old policy,
        the entropy of the current policy, and the ratio between the current policy and the old
        policy.

        The loss of the cost performance is defined as:

        .. math::
            L = \underset{a \sim \pi_{\theta^{old}}}{\mathbb{E}}[\lambda \frac{1 - \gamma \nu}{1 - \gamma}
            \frac{\pi_\theta(a|s)}{\pi_\theta^{old}(a|s)} A^{C}_{\pi_{\theta}^{old}}
            + KL(\pi_\theta(a|s)||\pi_\theta^{old}(a|s))]

        where :math:`\lambda` is the Lagrange multiplier,
        :math:`\frac{1 - \gamma \nu}{1 - \gamma}` is the coefficient value,
        :math:`\pi_\theta(a_t|s_t)` is the current policy,
        :math:`\pi_\theta^{old}(a_t|s_t)` is the old policy,
        :math:`A^{C}_{\pi_{\theta}^{old}}` is the cost advantage,
        :math:`KL(\pi_\theta(a_t|s_t)||\pi_\theta^{old}(a_t|s_t))` is the KL divergence between the
        current policy and the old policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            cost_adv (torch.Tensor): Cost advantage.
        """
        dist, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)

        coef = (1 - self.cfgs.buffer_cfgs.gamma * self.cfgs.buffer_cfgs.lam) / (
            1 - self.cfgs.buffer_cfgs.gamma
        )
        cost_loss = (self.lagrangian_multiplier * coef * ratio * cost_adv + kl_new_old).mean()
        self.loss_record.append(loss_pi_c=cost_loss.item())

        # useful extra info
        temp_max = torch.max(ratio).detach().mean().item()
        temp_min = torch.min(ratio).detach().mean().item()
        if temp_max > self.max_ratio:
            self.max_ratio = temp_max
        if temp_min < self.min_ratio:
            self.min_ratio = temp_min
        approx_kl = 0.5 * (log_p - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = {'kl': approx_kl, 'ent': ent, 'ratio': ratio.mean().item()}

        return cost_loss, pi_info

    # pylint: disable-next=too-many-locals
    def update(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Update actor, critic, running statistics as we used in the :class:`PolicyGradient`.

        In addition, we also update the Lagrange multiplier parameter,
        by calling the :meth:`update_lagrange_multiplier` function.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self.update_lagrange_multiplier(Jc)
        # the first stage is to maximize reward.
        data = PPO.update(self)
        # the second stage is to minimize cost.
        # get the loss before
        loss_pi_c_before = self.loss_record.get_mean('loss_pi_c')
        self.loss_record.reset('loss_pi_c')
        obs, act, log_p, cost_adv = (
            data['obs'],
            data['act'],
            data['log_p'],
            data['cost_adv'],
        )
        with torch.no_grad():
            old_dist = self.actor_critic.actor(obs)
            old_mean, old_std = old_dist.mean, old_dist.stddev
        # load the data into the data loader.
        dataset = torch.utils.data.TensorDataset(obs, act, log_p, cost_adv, old_mean, old_std)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfgs.num_mini_batches, shuffle=True
        )

        # update the policy net several times
        for i in range(self.cfgs.actor_iters):
            for _, (obs_b, act_b, log_p_b, cost_adv_b, old_mean_b, old_std_b) in enumerate(loader):
                # compute the old distribution of policy net.
                self.p_dist = torch.distributions.Normal(old_mean_b, old_std_b)
                # compute the loss of cost performance.
                loss_pi_c, pi_info_c = self.compute_loss_cost_performance(
                    obs_b, act_b, log_p_b, cost_adv_b
                )
                # update the policy net.
                self.actor_optimizer.zero_grad()
                # backward
                loss_pi_c.backward()
                # clip the gradient of policy net.
                if self.cfgs.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
                    )
                # average the gradient of policy net.
                distributed_utils.mpi_avg_grads(self.actor_critic.actor)
                self.actor_optimizer.step()
            # compute the new distribution of policy net.
            new_dist = self.actor_critic.actor(obs)
            # compute the KL divergence between old and new distribution.
            torch_kl = (
                torch.distributions.kl.kl_divergence(old_dist, new_dist)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            # if the KL divergence is larger than the target KL divergence, stop the update.
            if self.cfgs.kl_early_stopping and torch_kl > self.cfgs.target_kl:
                self.logger.log(f'KL early stop at the {i+1} th step in the second stage.')
                break

        loss_pi_c = self.loss_record.get_mean('loss_pi_c')
        # log the information.
        self.logger.store(
            **{
                'Loss/Loss_pi_c': loss_pi_c,
                'Loss/Delta_loss_pi_c': loss_pi_c - loss_pi_c_before,
                'Train/SecondStepStopIter': i + 1,
                'Train/SecondStepEntropy': pi_info_c['ent'],
                'Train/SecondStepPolicyRatio': pi_info_c['ratio'],
            }
        )
        return data
