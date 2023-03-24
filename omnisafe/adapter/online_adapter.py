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
"""Online Adapter for OmniSafe."""

from typing import Dict, Tuple

import torch

from omnisafe.envs.core import make, support_envs
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config


class OnlineAdapter:
    """Online Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._env_id = env_id
        self._env = make(env_id, num_envs=num_envs)
        self._test_env = make(env_id, num_envs=1)
        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)
        self._test_env.set_seed(seed)

        self._cfgs = cfgs
        self._device = cfgs.train_cfgs.device

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ):
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, time_limit=1000)
            self._test_env = TimeLimit(self._test_env, time_limit=1000)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env)
            self._test_env = AutoReset(self._test_env)
        if obs_normalize:
            self._env = ObsNormalize(self._env)
            self._test_env = ObsNormalize(self._test_env)
        if reward_normalize:
            self._env = RewardNormalize(self._env)
        if cost_normalize:
            self._env = CostNormalize(self._env)
        self._env = ActionScale(self._env, low=-1.0, high=1.0)
        self._test_env = ActionScale(self._test_env, low=-1.0, high=1.0)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env)
        self._test_env = Unsqueeze(self._test_env)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment.

        Returns:
            OmnisafeSpace: the action space.
        """
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment.

        Returns:
            OmnisafeSpace: the observation space.
        """
        return self._env.observation_space

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended, in which case further step()
            calls will return undefined results.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        obs, reward, cost, terminated, truncated = map(
            lambda x: torch.as_tensor(x, dtype=torch.float32, device=self._device),
            (obs, reward, cost, terminated, truncated),
        )
        return obs, reward, cost, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Resets the environment and returns an initial observation.

        Args:
            seed (Optional[int]): seed for the environment.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset()
        return obs.to(self._device), info

    def save(self) -> Dict[str, torch.nn.Module]:
        """Save the environment.

        Returns:
            Dict[str, torch.nn.Module]: the saved environment.
        """
        return self._env.save()
