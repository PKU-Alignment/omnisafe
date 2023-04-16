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
"""Abstract base class for buffer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from gymnasium.spaces import Box

from omnisafe.typing import OmnisafeSpace, cpu


class BaseBuffer(ABC):
    """Abstract base class for buffer."""

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = cpu,
    ) -> None:
        """Initialize the buffer.

        .. warning::
            The buffer only supports Box spaces.

        In  base buffer, we store the following data:

        .. list-table::

            *   -   Name
                -   Shape
                -   Dtype
                -   Description
            *   -   obs
                -   (size, obs_space.shape)
                -   torch.float32
                -   The observation.
            *   -   act
                -   (size, act_space.shape)
                -   torch.float32
                -   The action.
            *   -   reward
                -   (size, )
                -   torch.float32
                -   Single step reward.
            *   -   cost
                -   (size, )
                -   torch.float32
                -   Single step cost.
            *   -   done
                -   (size, )
                -   torch.float32
                -   Whether the episode is done.

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            size (int): The size of the buffer.
            device (torch.device): The device of the buffer.

        Attributes:
            data (dict[str, torch.Tensor]): The data of the buffer.
            _size (int): The size of the buffer.
            _device (torch.device): The device of the buffer.

        """
        self._device = device
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError
        if isinstance(act_space, Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

        self.data: dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }
        self._size = size

    @property
    def device(self) -> torch.device:
        """Return the device of the buffer."""
        return self._device

    @property
    def size(self) -> int:
        """Return the size of the buffer."""
        return self._size

    def __len__(self) -> int:
        """Return the length of the buffer."""
        return self._size

    def add_field(self, name: str, shape: tuple, dtype: torch.dtype):
        """Add a field to the buffer.

        Example:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    @abstractmethod
    def store(self, **data: torch.Tensor):
        """Store a transition in the buffer.

        .. warning::
            This is an abstract method.

        Example:
            >>> buffer = BaseBuffer(...)
            >>> buffer.store(obs=obs, act=act, reward=reward, cost=cost, done=done)

        Args:
            data (torch.Tensor): The data to store.
        """
