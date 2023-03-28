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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

import torch

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.utils.config import Config


class EarlyTerminatedAdapter(OnPolicyAdapter):
    """OnPolicy Adapter for OmniSafe."""

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        assert num_envs == 1, 'EarlyTerminatedAdapter only supports num_envs=1.'

        super().__init__(env_id, num_envs, seed, cfgs)

        self._cost_limit = cfgs.cost_limit
        self._cost_logger = torch.zeros(self._env.num_envs)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        next_obs, reward, cost, terminated, truncated, info = super().step(action)

        self._cost_logger += info.get('original_cost', cost)

        if self._cost_logger > self._cost_limit:
            reward = torch.zeros(self._env.num_envs)  # r_e = 0
            terminated = torch.ones(self._env.num_envs)
            next_obs, _ = self._env.reset()
            self._cost_logger = torch.zeros(self._env.num_envs)

        return next_obs, reward, cost, terminated, truncated, info
