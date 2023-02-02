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
"""Implementation of the FOCOPS algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange


@registry.register
class FOCOPS(PolicyGradient, Lagrange):
    """The First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize FOCOPS.

        FOCOPS is a combination of :class:`PolicyGradient` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        PolicyGradient.__init__(
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
        self.p_dist = None

    def algorithm_specific_logs(self) -> None:
        """Log the FOCOPS specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    # pylint: disable-next=too-many-arguments
    def compute_loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Computing pi/actor loss.
        In FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
            L = \nabla_\theta D_{K L}\left(\pi_\theta \| \pi_{\theta^{old}}\right)[s]
            -\frac{1}{\eta} \underset{a \sim \pi_{\theta^{old}}}
            {\mathbb{E}}\left[\frac{\nabla_\theta \pi_\theta(a \mid s)}
            {\pi_{\theta^{old}}(a \mid s)}\left(A^{R}_{\pi_{\theta^{old}}}(s, a)
            -\lambda A^C_{\pi_{\theta^{old}}}(s, a)\right)\right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta_k}` is the current policy.
        """
        dist, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)
        loss_pi = (kl_new_old - (1 / self.lam) * ratio * adv) * (
            kl_new_old.detach() <= self.eta
        ).type(torch.float32)
        loss_pi = loss_pi.mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        # useful extra info
        approx_kl = 0.5 * (log_p - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = {'kl': approx_kl, 'ent': ent, 'ratio': ratio.mean().item()}

        return loss_pi, pi_info

    def compute_surrogate(
        self,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv (torch.Tensor): reward advantage
            cost_adv (torch.Tensor): cost advantage
        """
        return (adv - self.lagrangian_multiplier * cost_adv) / (1 + self.lagrangian_multiplier)

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
        data = self.buf.get()
        obs, act, log_p, target_v, target_c, adv, cost_adv = (
            data['obs'],
            data['act'],
            data['log_p'],
            data['target_v'],
            data['target_c'],
            data['adv'],
            data['cost_adv'],
        )
        # get the loss before
        loss_pi_before, loss_v_before = self.loss_record.get_mean('loss_pi', 'loss_v')
        if self.cfgs.use_cost:
            loss_c_before = self.loss_record.get_mean('loss_c')
        self.loss_record.reset('loss_pi', 'loss_v', 'loss_c')
        with torch.no_grad():
            old_dist = self.actor_critic.actor(obs)
            old_mean, old_std = old_dist.mean, old_dist.stddev

        # load the data into the data loader.
        dataset = torch.utils.data.TensorDataset(
            obs, act, target_v, target_c, log_p, adv, cost_adv, old_mean, old_std
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfgs.num_mini_batches, shuffle=True
        )

        # update the value net, cost net and policy net for several times.
        for i in range(self.cfgs.actor_iters):
            for _, (
                obs_b,
                act_b,
                target_v_b,
                target_c_b,
                log_p_b,
                adv_b,
                cost_adv_b,
                old_mean_b,
                old_std_b,
            ) in enumerate(loader):
                # update the value net.
                self.update_value_net(obs_b, target_v_b)
                # update the cost net, if use cost.
                if self.cfgs.use_cost:
                    self.update_cost_net(obs_b, target_c_b)
                # update the policy net.
                self.p_dist = torch.distributions.Normal(old_mean_b, old_std_b)
                self.update_policy_net(obs_b, act_b, log_p_b, adv_b, cost_adv_b)
            # compute the new distribution of policy net.
            new_dist = self.actor_critic.actor(obs)
            # compute the KL divergence between old and new distribution.
            torch_kl = (
                torch.distributions.kl.kl_divergence(old_dist, new_dist)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            # if the KL divergence is larger than the target KL divergence, stop the update.
            if self.cfgs.kl_early_stopping and torch_kl > self.cfgs.target_kl:
                self.logger.log(f'KL early stop at the {i+1} th step.')
                break
        # log the information.
        loss_pi, loss_v = self.loss_record.get_mean('loss_pi', 'loss_v')
        self.logger.store(
            **{
                'Loss/Loss_pi': loss_pi,
                'Loss/Delta_loss_pi': loss_pi - loss_pi_before,
                'Train/StopIter': i + 1,
                'Values/Adv': adv.mean().item(),
                'Train/KL': torch_kl,
                'Loss/Delta_loss_reward_critic': loss_v - loss_v_before,
                'Loss/Loss_reward_critic': loss_v,
            }
        )
        if self.cfgs.use_cost:
            loss_c = self.loss_record.get_mean('loss_c')
            self.logger.store(
                **{
                    'Loss/Delta_loss_cost_critic': loss_c - loss_c_before,
                    'Loss/Loss_cost_critic': loss_c,
                }
            )
        return data
