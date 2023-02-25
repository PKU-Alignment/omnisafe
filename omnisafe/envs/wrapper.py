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


from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, Wrapper


class TimeLimit(Wrapper):
    """Time limit wrapper for the environment.

    Example:
        >>> env = TimeLimit(env, time_limit=100)
    """

    def __init__(self, env: CMDP, time_limit: int) -> None:
        """Initialize the time limit wrapper.

        Args:
            env (CMDP): The environment to wrap.
            time_limit (int): The time limit for each episode.
        """
        super().__init__(env)
        self._time_limit: int = time_limit
        self._time: Union[int, np.ndarray] = (
            0 if self.num_envs == 1 else np.array([0] * self.num_envs)
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        self._time = 0 if self.num_envs == 1 else np.array([0] * self.num_envs)
        return super().reset(seed)

    def single_reset(self, idx: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        if isinstance(self._time, np.ndarray):
            self._time[idx] = 0
        else:
            self._time = 0
        return super().single_reset(idx, seed)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)

        self._time += 1
        truncated = torch.tensor(self._time >= self._time_limit, dtype=torch.bool)

        return obs, reward, cost, terminated, truncated, info


class AutoReset(Wrapper):
    """Auto reset the environment when the episode is terminated.

    Example:
        >>> env = AutoReset(env)

    """

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if self.num_envs == 1:
            if terminated or truncated:
                obs, _ = self.reset()
        else:
            dones = terminated | truncated
            for idx, done in enumerate(dones):
                if done:
                    obs[idx], _ = self.single_reset(idx)

        return obs, reward, cost, terminated, truncated, info


class ObsNormalize(Wrapper):
    """Normalize the observation.

    Example:
        >>> env = ObsNormalize(env)

        >>> norm = Normalizer(env.observation_space.shape) # load saved normalizer
        >>> env = ObsNormalize(env, norm)

    """

    def __init__(self, env: CMDP, norm: Optional[Normalizer] = None) -> None:
        super().__init__(env)
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

        if norm is not None:
            self._obs_normalizer = norm
        else:
            self._obs_normalizer = Normalizer(self.observation_space.shape, clip=5)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        obs, info = super().reset(seed)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, info

    def single_reset(self, idx: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        obs, info = super().single_reset(idx, seed)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs.unsqueeze(0)).squeeze(0)
        return obs, info

    def save(self) -> Dict[str, torch.nn.Module]:
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

    def __init__(self, env: CMDP, norm: Optional[Normalizer] = None) -> None:
        """Initialize the reward normalizer.

        Args:
            env (CMDP): The environment to wrap.
            norm (Optional[Normalizer], optional): The normalizer to use. Defaults to None.

        """
        super().__init__(env)
        if norm is not None:
            self._reward_normalizer = norm
        else:
            self._reward_normalizer = Normalizer((), clip=5)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_reward'] = reward
        reward = self._reward_normalizer.normalize(reward)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> Dict[str, torch.nn.Module]:
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

    def __init__(self, env: CMDP, norm: Optional[Normalizer] = None) -> None:
        """Initialize the cost normalizer.

        Args:
            env (CMDP): The environment to wrap.
            norm (Normalizer, optional): The normalizer to use. Defaults to None.
        """
        super().__init__(env)
        if norm is not None:
            self._obs_normalizer = norm
        else:
            self._cost_normalizer = Normalizer((), clip=5)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_cost'] = cost
        cost = self._cost_normalizer.normalize(cost)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> Dict[str, torch.nn.Module]:
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
        low: Union[int, float],
        high: Union[int, float],
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            low: The lower bound of the action space.
            high: The upper bound of the action space.
        """
        super().__init__(env)
        assert isinstance(self.action_space, spaces.Box), 'Action space must be Box'

        self._old_min_action = torch.tensor(self.action_space.low, dtype=torch.float32)
        self._old_max_action = torch.tensor(self.action_space.high, dtype=torch.float32)

        min_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + low
        max_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + high
        self._action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=self.action_space.shape,
            dtype=self.action_space.dtype,  # type: ignore
        )

        self._min_action = torch.tensor(min_action, dtype=torch.float32)
        self._max_action = torch.tensor(max_action, dtype=torch.float32)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        action = self._old_min_action + (self._old_max_action - self._old_min_action) * (
            action - self._min_action
        ) / (self._max_action - self._min_action)
        return super().step(action)


class Unsqueeze(Wrapper):
    """Unsqueeze the observation, reward, cost, terminated, truncated and info.

    Example:
        >>> env = Unsqueeze(env)
    """

    def __init__(self, env: CMDP) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        assert self.num_envs == 1, 'Unsqueeze only works with single environment'
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = map(
            lambda x: x.unsqueeze(0), (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        obs, info = super().reset(seed)
        obs = obs.unsqueeze(0)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, info
