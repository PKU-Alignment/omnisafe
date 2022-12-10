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

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange


@registry.register
class FOCOPS(PolicyGradient, Lagrange):
    """The First Order Constrained Optimization in Policy Space(FOCOPS) algorithm.

    References:
        Paper Name: First Order Constrained Optimization in Policy Space.
        Paper author: Yiming Zhang, Quan Vuong, Keith W. Ross.
        Paper URL: https://arxiv.org/abs/2002.06506

    """

    def __init__(
        self,
        env,
        cfgs,
        algo='FOCOPS',
    ):
        """init."""

        PolicyGradient.__init__(
            self,
            env=env,
            cfgs=cfgs,
            algo=algo,
        )

        Lagrange.__init__(
            self,
            cost_limit=self.cfgs.lagrange_cfgs.cost_limit,
            lagrangian_multiplier_init=self.cfgs.lagrange_cfgs.lagrangian_multiplier_init,
            lambda_lr=self.cfgs.lagrange_cfgs.lambda_lr,
            lambda_optimizer=self.cfgs.lagrange_cfgs.lambda_optimizer,
        )
        self.lagrangian_multiplier = 0.0
        self.lam = self.cfgs.lam
        self.eta = self.cfgs.eta

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier)

    def update_lagrange_multiplier(self, Jc):
        """Update Lagrange multiplier."""
        self.lagrangian_multiplier += self.lambda_lr * (Jc - self.cost_limit)
        if self.lagrangian_multiplier < 0.0:
            self.lagrangian_multiplier = 0.0
        elif self.lagrangian_multiplier > 2.0:
            self.lagrangian_multiplier = 2.0

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)
        loss_pi = (
            kl_new_old
            - (1 / self.lam) * ratio * (data['adv'] - self.lagrangian_multiplier * data['cost_adv'])
        ) * (kl_new_old.detach() <= self.eta).type(torch.float32)
        loss_pi = loss_pi.mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """Update."""
        raw_data, data = self.buf.pre_process_data()
        # First update Lagrange multiplier parameter
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        self.update_lagrange_multiplier(Jc)
        # Then update policy network
        self.update_policy_net(data=data)
        # Update value network
        self.update_value_net(data=data)
        # Update cost network
        self.update_cost_net(data=data)
        return raw_data, data
