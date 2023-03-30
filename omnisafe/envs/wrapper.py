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
"""Wrapper for the environment."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, Wrapper


class TimeLimit(Wrapper):
    """Time limit wrapper for the environment.

    Example:
        >>> env = TimeLimit(env, time_limit=100)

    Attributes:
        _time_limit (int): The time limit for each episode.
        _time (int): The current time step.
    """

    def __init__(self, env: CMDP, time_limit: int, device: torch.device) -> None:
        """Initialize the time limit wrapper.

        .. warning::
            The time limit wrapper only supports single environment.

        Args:
            env (CMDP): The environment to wrap.
            time_limit (int): The time limit for each episode.
        """
        super().__init__(env=env, device=device)

        assert self.num_envs == 1, 'TimeLimit only supports single environment'

        self._time_limit: int = time_limit
        self._time: int = 0

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        .. note::
            Additionally, the time step will be reset to 0.

        Args:
            seed (int, optional): The seed for the environment. Defaults to None.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._time = 0
        return super().reset(seed)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            Additionally, the time step will be increased by 1.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        self._time += 1
        truncated = torch.tensor(
            self._time >= self._time_limit,
            dtype=torch.bool,
            device=self._device,
        )

        return obs, reward, cost, terminated, truncated, info


class AutoReset(Wrapper):
    """Auto reset the environment when the episode is terminated.

    Example:
        >>> env = AutoReset(env)

    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        super().__init__(env=env, device=device)

        assert self.num_envs == 1, 'AutoReset only supports single environment'

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            If the episode is terminated, the environment will be reset.
            The ``obs`` will be the first observation of the new episode.
            And the true final observation will be stored in ``info['final_observation']``.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            new_obs, new_info = self.reset()
            assert (
                'final_observation' not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs = new_obs
            info = new_info

        return obs, reward, cost, terminated, truncated, info


class ObsNormalize(Wrapper):
    """Normalize the observation.

    Example:
        >>> env = ObsNormalize(env)
        >>> norm = Normalizer(env.observation_space.shape) # load saved normalizer
        >>> env = ObsNormalize(env, norm)

    Attributes:
        _obs_normalizer (Normalizer): The normalizer for the observation.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        super().__init__(env=env, device=device)
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

        if norm is not None:
            self._obs_normalizer = norm.to(self._device)
        else:
            self._obs_normalizer = Normalizer(self.observation_space.shape, clip=5).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The observation and the ``info['final_observation']`` will be normalized.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'final_observation' in info:
            final_obs_slice = info['_final_observation'] if self.num_envs > 1 else slice(None)
            info['final_observation'] = info['final_observation'].to(self._device)
            info['original_final_observation'] = info['final_observation']
            info['final_observation'][final_obs_slice] = self._obs_normalizer.normalize(
                info['final_observation'][final_obs_slice],
            )
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Resets the environment and returns an initial observation.

        Args:
            seed (Optional[int]): seed for the environment.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = super().reset(seed)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the normalizer.

        .. note::
            When evaluating the saved model, the normalizer should be loaded.

        Returns:
            dict[str, torch.nn.Module]: The saved normalizer.
        """
        saved = super().save()
        saved['obs_normalizer'] = self._obs_normalizer
        return saved


class RewardNormalize(Wrapper):
    """Normalize the reward.

    Example:
        >>> env = RewardNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = RewardNormalize(env, norm)

    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize the reward normalizer.

        Args:
            env (CMDP): The environment to wrap.
            norm (Optional[Normalizer], optional): The normalizer to use. Defaults to None.

        """
        super().__init__(env=env, device=device)
        if norm is not None:
            self._reward_normalizer = norm.to(self._device)
        else:
            self._reward_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.


        .. note::
            The reward will be normalized for agent training.
            Then the original reward will be stored in ``info['original_reward']`` for logging.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_reward'] = reward
        reward = self._reward_normalizer.normalize(reward)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the normalizer.

        Returns:
            dict[str, torch.nn.Module]: The saved normalizer.
        """
        saved = super().save()
        saved['reward_normalizer'] = self._reward_normalizer
        return saved


class CostNormalize(Wrapper):
    """Normalize the cost.

    Example:
        >>> env = CostNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = CostNormalize(env, norm)
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize the cost normalizer.

        Args:
            env (CMDP): The environment to wrap.
            norm (Normalizer, optional): The normalizer to use. Defaults to None.
        """
        super().__init__(env=env, device=device)
        if norm is not None:
            self._obs_normalizer = norm.to(self._device)
        else:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The cost will be normalized for agent training.
            Then the original reward will be stored in ``info['original_cost']`` for logging.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_cost'] = cost
        cost = self._cost_normalizer.normalize(cost)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        saved = super().save()
        saved['cost_normalizer'] = self._cost_normalizer
        return saved


class ActionScale(Wrapper):
    """Scale the action space to a given range.

    Example:
        >>> env = ActionScale(env, low=-1, high=1)
        >>> env.action_space
        Box(-1.0, 1.0, (1,), float32)
    """

    def __init__(
        self,
        env: CMDP,
        device: torch.device,
        low: int | float,
        high: int | float,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            low: The lower bound of the action space.
            high: The upper bound of the action space.
        """
        super().__init__(env=env, device=device)
        assert isinstance(self.action_space, spaces.Box), 'Action space must be Box'

        self._old_min_action = torch.tensor(
            self.action_space.low,
            dtype=torch.float32,
            device=self._device,
        )
        self._old_max_action = torch.tensor(
            self.action_space.high,
            dtype=torch.float32,
            device=self._device,
        )

        min_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + low
        max_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + high
        self._action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=self.action_space.shape,
            dtype=self.action_space.dtype,  # type: ignore
        )

        self._min_action = torch.tensor(min_action, dtype=torch.float32, device=self._device)
        self._max_action = torch.tensor(max_action, dtype=torch.float32, device=self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The action will be scaled to the original range for agent training.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        action = self._old_min_action + (self._old_max_action - self._old_min_action) * (
            action - self._min_action
        ) / (self._max_action - self._min_action)
        return super().step(action)


class Unsqueeze(Wrapper):
    """Unsqueeze the observation, reward, cost, terminated, truncated and info.

    Example:
        >>> env = Unsqueeze(env)
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            device: The device to use.
        """
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'Unsqueeze only works with single environment'
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        action = action.squeeze(0)
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = (
            x.unsqueeze(0) for x in (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Resets the environment and returns a new observation.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            seed (int): The seed to use for the environment.

        Returns:
            observation (torch.Tensor): The initial observation of the space. Initial reward is assumed to be 0.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = super().reset(seed)
        obs = obs.unsqueeze(0)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, info
