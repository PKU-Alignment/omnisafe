# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Example and template for environment customization."""

from __future__ import annotations

import random
from typing import Any, ClassVar

import torch
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register


# first, define the environment class.
# the most important thing is to add the `env_register` decorator.
@env_register
class CustomExampleEnv(CMDP):

    # define what tasks the environment support.
    _support_envs: ClassVar[list[str]] = ['Custom-v0']

    # automatically reset when `terminated` or `truncated`
    need_auto_reset_wrapper = True
    # set `truncated=True` when the total steps exceed the time limit.
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        self._count = 0
        self._num_envs = 1
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs = torch.as_tensor(self._observation_space.sample())
        reward = 2 * torch.as_tensor(random.random())  # noqa
        cost = 2 * torch.as_tensor(random.random())  # noqa
        terminated = torch.as_tensor(random.random() > 0.9)  # noqa
        truncated = torch.as_tensor(self._count > self.max_episode_steps)
        return obs, reward, cost, terminated, truncated, {'final_observation': obs}

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return 10

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self.set_seed(seed)
        obs = torch.as_tensor(self._observation_space.sample())
        self._count = 0
        return obs, {}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def close(self) -> None:
        pass

    def render(self) -> Any:
        pass


# Then you can use it like this:
agent = omnisafe.Agent(
    'PPOLag',
    'Custom-v0',
)
agent.learn()
