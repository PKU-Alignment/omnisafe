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
"""Implementation of IPO algorithm."""

from typing import NamedTuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class IPO(PPO):
    """The Implementation of the IPO algorithm.

    References:
        - Title: IPO: Interior-point Policy Optimization under Constraints
        - Authors: Yongshuai Liu, Jiaxin Ding, Xin Liu.
        - URL: `IPO <https://arxiv.org/pdf/1910.09615.pdf>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize IPO."""
        PPO.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        self.penalty = 0

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Penalty', self.penalty)

    def compute_surrogate(self, adv: torch.Tensor, cost_adv: torch.Tensor) -> torch.Tensor:
        """Compute surrogate loss."""
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        self.penalty = self.cfgs.kappa / (self.cfgs.cost_limit - Jc + 1e-8)
        if self.penalty < 0 or self.penalty > self.cfgs.penalty_max:
            self.penalty = self.cfgs.penalty_max
        return (adv - self.penalty * cost_adv) / (1 + self.penalty)
