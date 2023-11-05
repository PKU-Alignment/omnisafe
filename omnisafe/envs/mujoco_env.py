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
"""Environments in the Safety Gymnasium."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box


@env_register
class MujocoEnv(CMDP):
    """Gymnasium Mujoco environment.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
        need_action_repeat_wrapper (bool): Whether to use action repeat wrapper.
        need_action_scale_wrapper (bool): Whether to use action scale wrapper.
    """

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    need_action_repeat_wrapper = True
    need_action_scale_wrapper = True

    _support_envs: ClassVar[list[str]] = [
        'Ant-v4',
        'Hopper-v4',
        'Walker2d-v4',
        'Humanoid-v4',
        'Swimmer-v4',
        'HalfCheetah-v4',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.

        Keyword Args:
            render_mode (str, optional): The render mode, ranging from ``human``, ``rgb_array``, ``rgb_array_list``.
                Defaults to ``rgb_array``.
            camera_name (str, optional): The camera name.
            camera_id (int, optional): The camera id.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.
        """
        super().__init__(env_id)
        self._env_id = env_id
        if num_envs == 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            self._env = gymnasium.make(id=env_id, autoreset=False, **kwargs)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._device = torch.device(device)

        self._num_envs = num_envs
        self._metadata = self._env.metadata

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

        .. note::

            OmniSafe use auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode.
            And the true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            cost: Amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, terminated, truncated)
        )
        cost = terminated.float()
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
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action.

        Returns:
            A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
