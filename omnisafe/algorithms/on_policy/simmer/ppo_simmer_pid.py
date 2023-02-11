# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the PID version of the Simmer algorithm using PPO."""

from typing import NamedTuple

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class PPOSimmerPid(PPO):
    """The PID version of the Simmer algorithm implemented with PPO.

    References:
        - Title: Effects of Safety State Augmentation on Safe Exploration
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Jun Wang, Haitham Bou Ammar.
        - URL: `Simmer RL <https://arxiv.org/abs/2206.02675>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize PPOSimmerPid.

        PPOSimmerPid is a combination of :class:`PPO` and :class:`Simmer` environment wrapper.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)

    def _specific_init_logs(self):
        super()._specific_init_logs()
        self.logger.register_key('Metrics/EpBudget')
        self.logger.register_key('Metrics/SafetyBudget')
