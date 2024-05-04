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
"""Environments of Safe Isaac Gym in the Safety-Gymnasium."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU


ISAAC_GYM_AVAILABLE = True
try:
    from omnisafe.utils.isaac_gym_utils import make_isaac_gym_env
except ImportError:
    ISAAC_GYM_AVAILABLE = False


@env_register
class SafetyIsaacGymEnv(CMDP):
    """Safe Isaac Gym Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
        need_evaluation (bool): Whether to create an environment for evaluation.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False
    need_evaluation: bool = False

    _support_envs: ClassVar[list[str]] = [
        'ShadowHandCatchOver2UnderarmSafeFinger',
        'ShadowHandOverSafeFinger',
        'ShadowHandCatchOver2UnderarmSafeJoint',
        'ShadowHandOverSafeJoint',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`SafetyIsaacGymEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        if ISAAC_GYM_AVAILABLE:
            self._env = make_isaac_gym_env(env_id=env_id, device=device, num_envs=num_envs)
        else:
            raise ImportError(
                'Please install Isaac Gym to use Safe Isaac Gym!\
                \nMore details please refer to https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.',
            )
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self.need_evaluation = False

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
        """Step the environment.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        return self._env.reset()

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Isaac Gym does not support manually render."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
