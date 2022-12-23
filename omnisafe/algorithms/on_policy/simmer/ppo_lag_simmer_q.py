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
"""Implementation of the Q Simmer algorithm using PPOLag."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag


@registry.register
class PPOLagSimmerQ(PPOLag):
    """The Q Simmer algorithm implemented with PPOLag.

    References:
        Title: Effects of Safety State Augmentation on Safe Exploration
        Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Jun Wang, Haitham Bou Ammar.
        URL: https://arxiv.org/abs/2206.02675
    """

    def __init__(self, env_id, cfgs) -> None:
        """Initialize PPOLagSimmerQ algorithm."""
        super().__init__(env_id=env_id, cfgs=cfgs)

    def algorithm_specific_logs(self):
        """Log the algorithm specific metrics."""
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/EpBudget')
        self.logger.log_tabular('Metrics/SafetyBudget')
