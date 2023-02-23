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
"""Implementation of the early terminated algorithm using PPO."""

from omnisafe.adapter import EarlyTerminatedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class PPOEarlyTerminated(PPO):
    """The early terminated algorithm implemented with PPO.

    References:
        Title: Safe Exploration by Solving Early Terminated MDP
        Authors: Hao Sun, Ziping Xu, Meng Fang, Zhenghao Peng, Jiadong Guo, Bo Dai, Bolei Zhou.
        URL: `Safe Exploration by Solving Early Terminated MDP <https://arxiv.org/abs/2107.04200>`_
    """

    def _init_env(self) -> None:
        self._env = EarlyTerminatedAdapter(
            self._env_id, self._cfgs.num_envs, self._seed, self._cfgs
        )
        self._steps_per_epoch = self._cfgs.steps_per_epoch
