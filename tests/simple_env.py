# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Simplest environment for testing."""

from __future__ import annotations

import random
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace


@env_register
class SimpleEnv(CMDP):
    """Simplest environment for testing."""

    _support_envs: ClassVar[list[str]] = ['Simple-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 30}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    need_action_scale_wrapper: bool = False
    _num_envs = 1
    _coordinate_observation_space: OmnisafeSpace

    def __init__(self, env_id: str, **kwargs) -> None:
        self._count = 0
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._coordinate_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    @property
    def get_cost_from_obs_tensor(self) -> None:
        return None

    @property
    def coordinate_observation_space(self) -> OmnisafeSpace:
        return self._coordinate_observation_space

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs = torch.as_tensor(self._observation_space.sample())
        reward = 10000 * torch.as_tensor(random.random())
        cost = 10000 * torch.as_tensor(random.random())
        terminated = torch.as_tensor(random.random() > 0.9)
        truncated = torch.as_tensor(self._count > 10)
        return obs, reward, cost, terminated, truncated, {'final_observation': obs}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        obs = torch.as_tensor(self._observation_space.sample())
        self._count = 0
        return obs, {}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def sample_action(self) -> torch.Tensor:
        return torch.as_tensor(self._action_space.sample())

    def render(self) -> Any:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        pass
