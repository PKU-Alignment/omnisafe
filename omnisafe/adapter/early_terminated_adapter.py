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
"""EarlyTerminated Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.utils.config import Config


class EarlyTerminatedAdapter(OnPolicyAdapter):
    """EarlyTerminated Adapter for OmniSafe.

    The EarlyTerminated Adapter is used to adapt the environment to the early terminated training.
    The adapter will terminate the episode when the accumulated cost exceeds the cost limit.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`EarlyTerminatedAdapter`."""
        assert num_envs == 1, 'EarlyTerminatedAdapter only supports num_envs=1.'

        super().__init__(env_id, num_envs, seed, cfgs)

        self._cost_limit: float = cfgs.algo_cfgs.cost_limit
        self._cost_logger: torch.Tensor = torch.zeros(self._env.num_envs).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            Early terminated adapter will accumulate the cost and terminate the episode when the
            accumulated cost exceeds the cost limit.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        next_obs, reward, cost, terminated, truncated, info = super().step(action)

        self._cost_logger += info.get('original_cost', cost)

        if self._cost_logger > self._cost_limit:
            reward = torch.zeros(self._env.num_envs).to(self._device)
            terminated = torch.ones(self._env.num_envs).to(self._device)
            next_obs, _ = self._env.reset()
            self._cost_logger = torch.zeros(self._env.num_envs).to(self._device)

        return next_obs, reward, cost, terminated, truncated, info
