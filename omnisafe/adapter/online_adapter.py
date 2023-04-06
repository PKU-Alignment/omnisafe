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

from __future__ import annotations

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
    """Online Adapter for OmniSafe.

    Online Adapter is used to adapt the environment to the online training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.

    Attributes:
        _env_id (str): The environment id.
        _env (CMDP): The environment.
        _cfgs (Config): The configuration.
        _device (torch.device): The device.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize the online adapter.

        OmniSafe is a framework for safe reinforcement learning. It is designed to be
        compatible with any existing RL algorithms. The online adapter is used
        to adapt the environment to the framework.

        OmniSafe provides a set of adapters to adapt the environment to the framework.

        - OnPolicyAdapter: Adapt the environment to the on-policy framework.
        - OffPolicyAdapter: Adapt the environment to the off-policy framework.
        - SauteAdapter: Adapt the environment to the SAUTE framework.
        - SimmerAdapter: Adapt the environment to the SIMMER framework.

        Args:
            env_id (str): The environment id.
            num_envs (int): The number of environments.
            seed (int): The random seed.
            cfgs (Config): The configuration.
        """
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs = cfgs
        self._device = cfgs.train_cfgs.device

        self._env_id = env_id
        self._env = make(env_id, num_envs=num_envs, device=self._device)
        self._eval_env = make(env_id, num_envs=1, device=self._device)
        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)
        self._eval_env.set_seed(seed)

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ):
        """Wrapper the environment.

        .. hint::

            OmniSafe supports the following wrappers:

            .. list-table::

                *   -   Wrapper
                    -   Description
                *   -   TimeLimit
                    -   Limit the time steps of the environment.
                *   -   AutoReset
                    -   Reset the environment when the episode is done.
                *   -   ObsNormalize
                    -   Normalize the observation.
                *   -   RewardNormalize
                    -   Normalize the reward.
                *   -   CostNormalize
                    -   Normalize the cost.
                *   -   ActionScale
                    -   Scale the action.
                *   -   Unsqueeze
                    -   Unsqueeze the step result for single environment case.

        Args:
            obs_normalize (bool): Whether to normalize the observation.
            reward_normalize (bool): Whether to normalize the reward.
            cost_normalize (bool): Whether to normalize the cost.
        """
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, time_limit=1000, device=self._device)
            self._eval_env = TimeLimit(self._eval_env, time_limit=1000, device=self._device)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

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
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

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
        return self._env.step(action)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Resets the environment and returns an initial observation.

        Args:
            seed (Optional[int]): seed for the environment.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._env.reset()

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the environment.

        Returns:
            Dict[str, torch.nn.Module]: the saved environment.
        """
        return self._env.save()
