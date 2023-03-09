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
"""Implementation of OffPolicyBuffer."""

from typing import Dict

import torch
from gymnasium.spaces import Box

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import OmnisafeSpace


class OffPolicyBuffer(BaseBuffer):
    """A ReplayBuffer for off_policy Algorithms."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(obs_space, act_space, size, device)
        if isinstance(obs_space, Box):
            self.data['next_obs'] = torch.zeros(
                (size, *obs_space.shape), dtype=torch.float32, device=device
            )
        else:
            raise NotImplementedError

        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size

        assert (
            self._max_size > self._batch_size
        ), 'The size of the buffer must be larger than the batch size.'

    @property
    def max_size(self) -> int:
        """Return the maximum size of the buffer."""
        return self._max_size

    @property
    def batch_size(self) -> int:
        """Return the batch size of the buffer."""
        return self._batch_size

    def store(self, **data: torch.Tensor):
        """Store data into the buffer."""
        for key, value in data.items():
            self.data[key][self._ptr] = value
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer."""
        idxs = torch.randint(0, self._size, (self._batch_size,))
        return {key: value[idxs] for key, value in self.data.items()}, idxs
