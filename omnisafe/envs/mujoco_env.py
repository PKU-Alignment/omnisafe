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
"""Environments in the Safety Gymnasium."""


from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register


@env_register
class MujocoEnv(CMDP):
    """Gymnasium Mujoco environment.
    Attributes:
        _support_envs (list[str]): List of supported environments.
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    _support_envs = [
        'Ant-v4',
        'Hopper-v4',
        'Walker2d-v4',
        'Humanoid-v4',
        'Swimmer-v4',
        'HalfCheetah-v4',
    ]
    need_auto_reset_wrapper = False

    need_time_limit_wrapper = False
    need_action_repeat_wrapper = True

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = 'cpu',
        **kwargs,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.
            **kwargs: Other arguments.
        """
        super().__init__(env_id)
        self._env_id = env_id
        if num_envs > 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            self._env = gymnasium.vector.make(
                id=env_id,
                num_envs=num_envs,
                **kwargs,
            )
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            self._env = gymnasium.make(id=env_id, autoreset=False, **kwargs)
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        self._device = torch.device(device)

        self._num_envs = num_envs
        self._metadata = self._env.metadata

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step the environment.

        .. note::

            OmniSafe use auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode.
            And the true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device) for x in (obs, reward, terminated, truncated)
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

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """Reset the environment.

        Args:
            seed (int, optional): Seed to reset the environment. Defaults to None.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset(seed=seed)
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
            torch.Tensor: A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Any: Rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def get_cost_from_obs_tensor(self, input_obs: torch.Tensor) -> torch.Tensor:
        """Check if the observation violates the environment's constraints."""
        assert torch.is_tensor(input_obs), 'obs must be tensor'
        if len(input_obs.shape) == 2:
            batch_size = input_obs.shape[0]
            obs = input_obs.reshape(batch_size, -1)
        elif len(input_obs.shape) == 3:
            batch_size = input_obs.shape[0] * input_obs.shape[1]
            obs = input_obs.reshape(batch_size, -1)

        if self._env_id == 'Ant-v4':
            min_z, max_z = 0.2, 1.0
            is_finite = torch.isfinite(obs).all()
            is_between = torch.logical_and(min_z < obs[:, 0], obs[:, 0] < max_z)
            is_healthy = torch.logical_and(is_finite, is_between)
        elif self._env_id == 'Humanoid-v4':
            min_z, max_z = 1.0, 2.0
            is_healthy = torch.logical_and(min_z < obs[:, 0], obs[:, 0] < max_z)
        elif self._env_id == 'Hopper-v4':
            z, angle = obs[:, 0:2]
            state = obs[:, 1:]
            min_state, max_state = -100.0, 100.0
            min_z, max_z = (0.7, float('inf'))
            min_angle, max_angle = (-0.2, 0.2)
            healthy_state = torch.logical_and(min_state < state, state < max_state)
            healthy_z = torch.logical_and(min_z < z, z < max_z)
            healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
            is_healthy = torch.all(torch.stack([healthy_state, healthy_z, healthy_angle]), dim=0)
        elif self._env_id == 'walker2d-v4':
            z, angle = obs[:, 0:2]
            min_z, max_z = (0.8, 2)
            min_angle, max_angle = (-1, 1)
            healthy_z = torch.logical_and(min_z < z, z < max_z)
            healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
            is_healthy = torch.logical_and(healthy_z, healthy_angle)
        else :
            is_healthy = torch.ones(batch_size, dtype=torch.bool, device=input_obs.device)
        cost = ~is_healthy
        if len(input_obs.shape) == 2:
            cost = cost.reshape(input_obs.shape[0], 1)
        elif len(input_obs.shape) == 3:
            cost = cost.reshape(input_obs.shape[0], input_obs.shape[1], 1)
        return cost.float()
