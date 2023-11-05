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
"""Online Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.envs.core import CMDP, make, support_envs
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
from omnisafe.utils.tools import get_device


class OnlineAdapter:
    """Online Adapter for OmniSafe.

    OmniSafe is a framework for safe reinforcement learning. It is designed to be compatible with
    any existing RL algorithms. The online adapter is used to adapt the environment to the
    framework.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnlineAdapter`."""
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs: Config = cfgs
        self._device: torch.device = get_device(cfgs.train_cfgs.device)
        self._env_id: str = env_id
        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device)
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device)

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. hint::
            OmniSafe supports the following wrappers:

        +-----------------+--------------------------------------------------------+
        | Wrapper         | Description                                            |
        +=================+========================================================+
        | TimeLimit       | Limit the time steps of the environment.               |
        +-----------------+--------------------------------------------------------+
        | AutoReset       | Reset the environment when the episode is done.        |
        +-----------------+--------------------------------------------------------+
        | ObsNormalize    | Normalize the observation.                             |
        +-----------------+--------------------------------------------------------+
        | RewardNormalize | Normalize the reward.                                  |
        +-----------------+--------------------------------------------------------+
        | CostNormalize   | Normalize the cost.                                    |
        +-----------------+--------------------------------------------------------+
        | ActionScale     | Scale the action.                                      |
        +-----------------+--------------------------------------------------------+
        | Unsqueeze       | Unsqueeze the step result for single environment case. |
        +-----------------+--------------------------------------------------------+


        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            time_limit = (
                self._cfgs.train_cfgs.time_limit
                if hasattr(self._cfgs.train_cfgs, 'time_limit')
                else 1000
            )
            self._env = TimeLimit(self._env, time_limit=time_limit, device=self._device)
            self._eval_env = TimeLimit(self._eval_env, time_limit=time_limit, device=self._device)
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
        if self._env.need_action_scale_wrapper:
            self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
            self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._env.observation_space

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
        return self._env.step(action)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be an empty dict. common wrappers are
            ``obs_normalize``, ``reward_normalize``, and ``cost_normalize``.

        Returns:
            The saved components of environment, e.g., ``obs_normalizer``.
        """
        return self._env.save()
