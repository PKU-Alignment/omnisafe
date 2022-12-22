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
"""Implementation of the TRPO PID-Lagrange algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.pid_lagrange import PIDLagrangian


@registry.register
class TRPOPid(TRPO, PIDLagrangian):
    """The Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.

    References:
        Paper Name: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.
        Paper author: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        Paper URL: https://arxiv.org/abs/2007.03964

    """

    def __init__(
        self,
        env_id,
        cfgs,
    ):

        TRPO.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        PIDLagrangian.__init__(self, **self.cfgs.PID_cfgs._asdict())

        self.cost_limit = self.cfgs.cost_limit

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.cost_penalty)
        self.logger.log_tabular('PID/pid_Kp', self.pid_kp)
        self.logger.log_tabular('PID/pid_Ki', self.pid_ki)
        self.logger.log_tabular('PID/pid_Kd', self.pid_kd)

    def compute_loss_pi(self, data: dict):
        """compute loss for policy"""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        penalty = self.cost_penalty
        loss_pi += penalty * (ratio * data['cost_adv']).mean()
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """update policy"""
        raw_data, data = self.buf.pre_process_data()
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.pid_update(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        return raw_data, data
