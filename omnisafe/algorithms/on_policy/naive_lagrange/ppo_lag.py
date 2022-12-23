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
"""Implementation of the Lagrange version of the PPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange


@registry.register
class PPOLag(PolicyGradient, Lagrange):
    """The Lagrange version of the PPO algorithm.

    References:
        Title: Benchmarking Safe Exploration in Deep Reinforcement Learning
        Authors: Alex Ray, Joshua Achiam, Dario Amodei.
        URL: https://cdn.openai.com/safexp-short.pdf
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env_id,
        cfgs,
    ):
        """Initialize PPO-Lag algorithm."""
        self.clip = cfgs.clip
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
        )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        """Compute policy loss."""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        loss_pi += self.cfgs.entropy_coef * dist.entropy().mean()

        # Ensure that Lagrange Multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += (
            penalty * (torch.max(ratio * data['cost_adv'], ratio_clip * data['cost_adv'])).mean()
        )
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """Update."""
        # pre-process data
        raw_data, data = self.buf.pre_process_data()
        # Note that logger already uses MPI statistics across all processes..
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(Jc)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        return raw_data, data
