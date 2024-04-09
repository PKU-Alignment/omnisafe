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

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace


@env_register
class CustomEnv(CMDP):
    """Simplest environment for the example and template for environment customization.

    If you wish for your environment to become part of the officially supported environments by
    OmniSafe, please refer to this document to implement environment embedding. We will welcome
    your GitHub pull request.

    Customizing the environment in OmniSafe requires specifying the following parameters:

    Attributes:
        _support_envs (ClassVar[list[str]]): A list composed of strings, used to display all task
            names supported by the customized environment. For example: ['Simple-v0'].
        _action_space: The action space of the task. It can be defined by directly passing an
            :class:`OmniSafeSpace` object, or specified in :meth:`__init__` based on the
            characteristics of the customized environment.
        _observation_space: The observation space of the task. It can be defined by directly
            passing an :class:`OmniSafeSpace` object, or specified in :meth:`__init__` based on the
            characteristics of the customized environment.
        metadata (ClassVar[dict[str, int]]): A class variable containing environment metadata, such
            as render FPS.
        need_time_limit_wrapper (bool): Whether the environment needs a time limit wrapper.
        need_auto_reset_wrapper (bool): Whether the environment needs an auto-reset wrapper.
        _num_envs (int): The number of parallel environments.

    .. warning::
        The :class:`omnisafe.adapter.OnlineAdapter`, :class:`omnisafe.adapter.OfflineAdapter`, and
        :class:`omnisafe.adapter.ModelBasedAdapter` implemented by OmniSafe use
        :class:`omnisafe.envs.wrapper.AutoReset` and :class:`omnisafe.envs.wrapper.TimeLimit` in
        algorithm updates. We recommend setting :attr:`need_auto_reset_wrapper` and
        :attr:`need_time_limit_wrapper` to ``True``. If you do not want to use these wrappers, you
        can add customized logic in the :meth:`step` function of the customized
        environment.
    """

    _support_envs: ClassVar[list[str]] = ['Simple-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    metadata: ClassVar[dict[str, int]] = {}
    env_spec_log: dict[str, Any]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(
        self,
        env_id: str,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Initialize CustomEnv with the given environment ID and optional keyword arguments.

        .. note::
            Optionally, you can specify some environment-specific information that needs to be
            logged. You need to complete this operation in two steps:

            1. Define the environment information in dictionary format in :meth:`__init__`.
            2. Log the environment information in :meth:`spec_log`. Please note that the logging in
                OmniSafe will occur at the end of each episode, so you need to consider how to
                reset the logging values for each episode.

        Example:
            >>> # First, define the environment information in dictionary format in __init__.
            >>> def __init__(self, env_id: str, **kwargs: Any) -> None:
            >>>     self.env_spec_log = {'Env/Interaction': 0,}
            >>>
            >>> # Then, log and reset the environment information in spec_log.
            >>> def spec_log(self, logger: Logger) -> dict[str, Any]:
            >>>     logger.store({'Env/Interaction': self.env_spec_log['Env/Interaction']})
            >>>     self.env_spec_log['Env/Interaction'] = 0

        Args:
            env_id (str): The environment ID.
            **kwargs: Additional keyword arguments.
        """
        self._count = 0
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._max_episode_steps = 10
        self.env_spec_log = {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            You need to implement dynamic features related to environment interaction here. That is:

            1. Update the environment state based on the action;
            2. Calculate reward and cost based on the environment state;
            3. Determine whether to terminate based on the environment state;
            4. Record the information you need.

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
        self._count += 1
        obs = torch.as_tensor(self._observation_space.sample())
        reward = 10000 * torch.as_tensor(random.random())  # noqa
        cost = 10000 * torch.as_tensor(random.random())  # noqa
        terminated = torch.as_tensor(random.random() > 0.9)  # noqa
        truncated = torch.as_tensor(self._count > self._max_episode_steps)
        return obs, reward, cost, terminated, truncated, {'final_observation': obs}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed to use for the environment. Defaults to None.
            options (dict[str, Any], optional): Additional options. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing:
                - obs (torch.Tensor): The initial observation.
                - info (dict): Additional information.
        """
        if seed is not None:
            self.set_seed(seed)
        obs = torch.as_tensor(self._observation_space.sample())
        self._count = 0
        return obs, {}

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return 10

    def spec_log(self, logger: Logger) -> None:
        """Log specific environment into logger.

        .. note::
            This function will be called after each episode.

        Args:
            logger (Logger): The logger to use for logging.
        """

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the environment.

        Args:
            seed (int): The random seed.
        """
        random.seed(seed)

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Any: An array representing the rendered environment.
        """
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
