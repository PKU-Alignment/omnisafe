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

from turtle import pen

import numpy as np
import torch

import omnisafe.algos.utils.distributed_tools as distributed_tools
from omnisafe.algos.common.lagrange import Lagrange
from omnisafe.algos.on_policy.policy_gradient import PolicyGradient
from omnisafe.algos.registry import REGISTRY


@REGISTRY.register
class FOCOPS(PolicyGradient, Lagrange):
    def __init__(self, algo='focops', **cfgs):

        PolicyGradient.__init__(self, algo=algo, **cfgs)

        Lagrange.__init__(self, **self.cfgs['lagrange_cfgs'])
        self.lagrangian_multiplier = 0.0
        self.lam = self.cfgs['lam']
        self.eta = self.cfgs['eta']

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier)

    def update_lagrange_multiplier(self, Jc):
        self.lagrangian_multiplier += self.lambda_lr * (Jc - self.cost_limit)
        if self.lagrangian_multiplier < 0.0:
            self.lagrangian_multiplier = 0.0
        elif self.lagrangian_multiplier > 2.0:
            self.lagrangian_multiplier = 2.0

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)
        loss_pi = (
            kl_new_old
            - (1 / self.lam) * ratio * (data['adv'] - self.lagrangian_multiplier * data['cost_adv'])
        ) * (kl_new_old.detach() <= self.eta).type(torch.float32)
        loss_pi = loss_pi.mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        data = self.pre_process_data(raw_data)
        # First update Lagrange multiplier parameter
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        self.update_lagrange_multiplier(ep_costs)
        # Then update policy network
        self.update_policy_net(data=data)
        # Update value network
        self.update_value_net(data=data)
        # Update cost network
        self.update_cost_net(data=data)
        self.update_running_statistics(raw_data)