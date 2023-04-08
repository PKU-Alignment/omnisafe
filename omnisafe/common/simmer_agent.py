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


class BaseSimmerAgent(ABC):
    """Base class for controlling safety budget of Simmer adapter."""

    def __init__(
        self,
        obs_space: tuple = (0),
        action_space: tuple = (-1, 1),
        history_len: int = 100,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Initialize the agent."""
        assert obs_space is not None, 'Please specify the state space for the Simmer agent'
        assert history_len > 0, 'History length should be positive'
        self._history_len = history_len
        self._obs_space = obs_space
        self._action_space = action_space
        # history
        self._error_history = deque([], maxlen=self._history_len)
        self._reward_history = deque([], maxlen=self._history_len)
        self._state_history = deque([], maxlen=self._history_len)
        self._action_history = deque([], maxlen=self._history_len)
        self._observation_history = deque([], maxlen=self._history_len)

    @abstractmethod
    def get_greedy_action(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the greedy action."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the action."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset the agent."""
        raise NotImplementedError


# pylint: disable-next=too-many-instance-attributes
class SimmerPIDAgent(BaseSimmerAgent):
    """Simmer PID agent."""

    def __init__(
        self,
        obs_space: tuple = (0),
        action_space: tuple = (-1, 1),
        history_len: int = 100,
        **kwargs,
    ) -> None:
        """Initialize the agent."""
        super().__init__(obs_space, action_space, history_len, **kwargs)
        self._kp = kwargs.get('kp', 0)
        self._ki = kwargs.get('ki', 0)
        self._kd = kwargs.get('kd', 0)
        self._polyak = kwargs.get('polyak', 0.995)
        self._budget_bound = kwargs.get('budget_bound', 25.0)
        self._sum_history = torch.zeros(1)
        self._prev_action = torch.zeros(1)
        self._prev_error = torch.zeros(1)
        self._prev_raw_action = torch.zeros(1)
        self._integral_history = deque([], maxlen=100)

    def get_greedy_action(
        self,
        safety_budget: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get the greedy action."""
        # compute the error
        current_error = safety_budget - observation
        # blur the error
        blured_error = self._polyak * self._prev_error + (1 - self._polyak) * current_error
        # log the history
        self._error_history.append(blured_error)
        # compute the integral
        self._integral_history.append(blured_error)
        self._sum_history = sum(self._integral_history)
        # proportional part
        p_part = self._kp * blured_error
        # integral part
        i_part = self._ki * self._sum_history
        # derivative part
        d_part = self._kd * (self._prev_action - self._prev_raw_action)
        # get the raw action
        raw_action = p_part + i_part + d_part
        # clip the action
        action = torch.clamp(raw_action, min=self._action_space[0], max=self._action_space[1])
        # get the next safety budget
        next_safety_budget = torch.clamp(safety_budget + action, 0.0, self._budget_bound)
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
        """Get the action."""
        return self.get_greedy_action(safety_budget, observation)

    def reset(self):
        """Resetting the internal state of the agent."""
        self._sum_history = torch.zeros(1)
        self._prev_action = torch.zeros(1)
        self._prev_error = torch.zeros(1)
        self._prev_raw_action = torch.zeros(1)
