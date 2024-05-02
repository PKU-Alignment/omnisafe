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
"""Implementation of OffPolicyBuffer."""

from __future__ import annotations

import torch
from gymnasium.spaces import Box

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class OffPolicyBuffer(BaseBuffer):
    r"""A ReplayBuffer for off_policy Algorithms.

    .. warning::
        The buffer only supports Box spaces.

    Compared to the base buffer, the off-policy buffer stores extra data:

    +----------+---------------------------+---------------+----------------------------------------+
    | Name     | Shape                     | Dtype         | Description                            |
    +==========+===========================+===============+========================================+
    | next_obs | (size, \*obs_space.shape) | torch.float32 | The next observation from environment. |
    +----------+---------------------------+---------------+----------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        penalty_coefficient (float, optional): The penalty coefficient. Defaults to 0.0.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data stored in the buffer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        penalty_coefficient: float = 0.0,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OffPolicyBuffer`."""
        super().__init__(obs_space, act_space, size, device)
        if isinstance(obs_space, Box):
            self.data['next_obs'] = torch.zeros(
                (size, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._penalty_coefficient: float = penalty_coefficient

        assert (
            self._max_size > self._batch_size
        ), 'The size of the buffer must be larger than the batch size.'

    @property
    def max_size(self) -> int:
        """Return the max size of the buffer."""
        return self._max_size

    @property
    def size(self) -> int:
        """Return the current size of the buffer."""
        return self._size

    @property
    def batch_size(self) -> int:
        """Return the batch size of the buffer."""
        return self._batch_size

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer.

        .. hint::
            The ReplayBuffer is a circular buffer. When the buffer is full, the oldest data will be
            overwritten.

        Args:
            data (torch.Tensor): The data to be stored.
        """
        for key, value in data.items():
            self.data[key][self._ptr] = value
        self.data['reward'][self._ptr] -= self._penalty_coefficient * self.data['cost'][self._ptr]
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        idxs = torch.randint(0, self._size, (self._batch_size,))
        return {key: value[idxs] for key, value in self.data.items()}
