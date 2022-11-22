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

from collections import deque, namedtuple

import torch

from omnisafe.algos.common.pid_lagrange import PID_Lagrangian
from omnisafe.algos.on_policy.policy_gradient import PolicyGradient
from omnisafe.algos.registry import REGISTRY


@REGISTRY.register
class CPPOPid(PolicyGradient, PID_Lagrangian):
    def __init__(self, algo: str = 'cppo-pid', **cfgs):

        PolicyGradient.__init__(self, algo=algo, **cfgs)
        PID_Lagrangian.__init__(self, **self.cfgs['PID_cfgs'])
        # self.update_params_from_local(locals())

        self.clip = self.cfgs['clip']
        self.cost_limit = self.cfgs['cost_limit']

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier', self.cost_penalty)
        self.logger.log_tabular('pid_Kp', self.pid_Kp)
        self.logger.log_tabular('pid_Ki', self.pid_Ki)
        self.logger.log_tabular('pid_Kd', self.pid_Kd)

    def compute_loss_pi(self, data: dict, **kwargs):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

        surr_adv = (torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        surr_cadv = (torch.max(ratio * data['cost_adv'], ratio_clip * data['cost_adv'])).mean()

        loss_pi = -surr_adv
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # ensure that lagrange multiplier is positive
        penalty = self.cost_penalty
        # loss_pi += penalty * ((ratio * data['cost_adv']).mean())
        loss_pi += penalty * surr_cadv
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('Metrics/EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.pid_update(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)
