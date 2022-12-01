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

import torch

from omnisafe.algos import registry
from omnisafe.algos.common.lagrange import Lagrange
from omnisafe.algos.on_policy.policy_gradient import PolicyGradient


@registry.register
class PDO(PolicyGradient, Lagrange):
    def __init__(self, **cfgs):

        PolicyGradient.__init__(self, **cfgs)
        cfgs = cfgs['cfgs']
        Lagrange.__init__(self, **self.cfgs['lagrange_cfgs'])
        self.max_grad_norm = cfgs['use_max_grad_norm']

    def compute_loss_pi(self, data: dict):
        """
        computing pi/actor loss

        Returns:
            torch.Tensor
        """
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        penalty_lambda = self.lambda_range_projection(self.lagrangian_multiplier).item()
        adv = data['adv'] - penalty_lambda * data['cost_adv']
        loss_pi = -(ratio * adv).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
