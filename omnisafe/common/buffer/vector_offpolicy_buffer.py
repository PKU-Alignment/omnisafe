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
"""Implementation of VectorOffPolicyBuffer."""

from __future__ import annotations

import torch
from gymnasium.spaces import Box

from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.typing import OmnisafeSpace, cpu


class VectorOffPolicyBuffer(OffPolicyBuffer):
    """A VectorReplayBuffer for OffPolicy Algorithms."""

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device = cpu,
    ) -> None:
        """Initialize the off policy buffer.

        The vector-off-policy buffer is a vectorized version of the off-policy buffer.
        It stores the data in a single tensor, and the data of each environment is
        stored in a separate column.

        .. warning::
            The buffer only supports Box spaces.

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            size (int): The size of the buffer.
            batch_size (int): The batch size of the buffer.
            num_envs (int): The number of environments.
            device (torch.device, optional): The device of the buffer. Defaults to
                torch.device('cpu').

        Attributes:
            data (Dict[str, torch.Tensor]): The data of the buffer.
            _ptr (int): The pointer of the buffer.
            _size (int): The size of the buffer.
            _max_size (int): The maximum size of the buffer.
            _batch_size (int): The batch size of the buffer.
            _num_envs (int): The number of environments.

        """
        self._num_envs = num_envs
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self.data = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'cost': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'done': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'next_obs': next_obs_buf,
        }

        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._device = device

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self._num_envs

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
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape),
            dtype=dtype,
            device=self._device,
        )

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch from the buffer."""
        idx = torch.randint(
            0,
            self._size,
            (self._batch_size * self._num_envs,),
            device=self._device,
        )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._batch_size)
        return {key: value[idx, env_idx] for key, value in self.data.items()}
