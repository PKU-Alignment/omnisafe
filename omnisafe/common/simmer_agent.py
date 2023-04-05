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

import torch
import numpy as np

from collections import deque

class BaseSimmerAgent(ABC):
    """"Base class for controlling safety budget of Simmer adapter."""
    def __init__(
        self,
        obs_space: list,
        action_space: list,
        history_len: int,
        **kwargs: dict, # pylint: disable=unused-argument
    ):
        """Initialize the agent."""
        assert obs_space is not None, "Please specify the state space for the Simmer agent"
        assert history_len > 0, "History length shoud be positive"
        assert type(action_space) == list, "entry action_space should be a list"
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
    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get the greedy action."""
        raise NotImplementedError
    
    @abstractmethod
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Get the action."""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset the agent."""
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Step the agent."""
        raise NotImplementedError
    
    @abstractmethod
    def reward_fn(self, observation: torch.Tensor, state: torch.Tensor, action: torch.Tensor):
        """Reward function."""
        raise NotImplementedError