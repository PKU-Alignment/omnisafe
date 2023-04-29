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
"""Implementation of Simmer agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import torch

from omnisafe.utils.config import Config


class BaseSimmerAgent(ABC):
    """Base class for controlling safety budget of Simmer adapter.

    Args:
        cfgs (Config): The configurations for the agent.
        budget_bound (torch.Tensor): The bound of the safety budget.
        obs_space (tuple[int,], optional): The observation space. Defaults to (0,).
        action_space (tuple[int, int], optional): The action space. Defaults to (-1, 1).
        history_len (int, optional): The length of the history. Defaults to 100.
    """

    def __init__(
        self,
        cfgs: Config,
        budget_bound: torch.Tensor,
        obs_space: tuple[int,] = (0,),
        action_space: tuple[int, int] = (-1, 1),
        history_len: int = 100,
    ) -> None:
        """Initialize an instance of :class:`BaseSimmerAgent`."""
        assert obs_space is not None, 'Please specify the state space for the Simmer agent'
        assert history_len > 0, 'History length should be positive'

        self._history_len: int = history_len
        self._obs_space: tuple[int,] = obs_space
        self._action_space: tuple[int, int] = action_space
        self._budget_bound: torch.Tensor = budget_bound
        self._cfgs: Config = cfgs
        # history
        self._error_history: deque[torch.Tensor] = deque([], maxlen=self._history_len)
        self._reward_history: deque[torch.Tensor] = deque([], maxlen=self._history_len)
        self._state_history: deque[torch.Tensor] = deque([], maxlen=self._history_len)
        self._action_history: deque[torch.Tensor] = deque([], maxlen=self._history_len)
        self._observation_history: deque[torch.Tensor] = deque([], maxlen=self._history_len)

    @abstractmethod
    def get_greedy_action(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the greedy action.

        Args:
            safety_budget (torch.Tensor): The current safety budget.
            observation (torch.Tensor): The current observation.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the action.

        Args:
            safety_budget (torch.Tensor): The current safety budget.
            observation (torch.Tensor): The current observation.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError


# pylint: disable-next=too-many-instance-attributes
class SimmerPIDAgent(BaseSimmerAgent):
    """Simmer PID agent.

    Args:
        cfgs (Config): The configurations for the agent.
        budget_bound (torch.Tensor): The bound of the safety budget.
        obs_space (tuple[int,], optional): The observation space. Defaults to (0,).
        action_space (tuple[int, int], optional): The action space. Defaults to (-1, 1).
        history_len (int, optional): The length of the history. Defaults to 100.
    """

    def __init__(
        self,
        cfgs: Config,
        budget_bound: torch.Tensor,
        obs_space: tuple[int,] = (0,),
        action_space: tuple[int, int] = (-1, 1),
        history_len: int = 100,
    ) -> None:
        """Initialize an instance of :class:`SimmerPIDAgent`."""
        super().__init__(
            cfgs=cfgs,
            budget_bound=budget_bound,
            obs_space=obs_space,
            action_space=action_space,
            history_len=history_len,
        )

        self._sum_history: torch.Tensor = torch.zeros(1)
        self._prev_action: torch.Tensor = torch.zeros(1)
        self._prev_error: torch.Tensor = torch.zeros(1)
        self._prev_raw_action: torch.Tensor = torch.zeros(1)
        self._integral_history: deque[torch.Tensor] = deque([], maxlen=10)

    def get_greedy_action(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the greedy action.

        Args:
            safety_budget (torch.Tensor): The current safety budget.
            observation (torch.Tensor): The current observation.

        Returns:
            The greedy action.
        """
        # compute the error
        current_error = safety_budget - observation
        # blur the error
        blured_error = (
            self._cfgs.polyak * self._prev_error + (1 - self._cfgs.polyak) * current_error
        )
        # log the history
        self._error_history.append(blured_error)
        # compute the integral
        self._integral_history.append(blured_error)
        self._sum_history = torch.as_tensor(sum(self._integral_history))
        # proportional part
        p_part = self._cfgs.kp * blured_error
        # integral part
        i_part = self._cfgs.ki * self._sum_history
        # derivative part
        d_part = self._cfgs.kd * (self._prev_action - self._prev_raw_action)
        # get the raw action
        raw_action = p_part + i_part + d_part
        # clip the action
        action = torch.clamp(raw_action, min=self._action_space[0], max=self._action_space[1])
        # get the next safety budget
        eps = 1e-6
        next_safety_budget = torch.clamp(
            safety_budget + action,
            eps * torch.ones_like(safety_budget),
            self._budget_bound,
        )
        # update the true action after clipping
        action = next_safety_budget - safety_budget
        # update the history
        self._prev_action, self._prev_raw_action, self._prev_error = (
            action,
            raw_action,
            blured_error,
        )

        return next_safety_budget

    def act(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the action.

        Args:
            safety_budget (torch.Tensor): The current safety budget.
            observation (torch.Tensor): The current observation.

        Returns:
            The selected action.
        """
        return self.get_greedy_action(safety_budget, observation)
