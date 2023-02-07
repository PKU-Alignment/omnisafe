# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Implementation of ReplayBuffer."""

from typing import Dict

import torch
from gymnasium.spaces import Box

from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.typing import OmnisafeSpace


class VectorOffPolicyBuffer(OffPolicyBuffer):
    """A ReplayBuffer for off_policy Algorithms."""

    def __init__(  # pylint: disable=super-init-not-called,  too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device = torch.device('cpu'),
    ):
        self._num_envs = num_envs
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape), dtype=torch.float32, device=device
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape), dtype=torch.float32, device=device
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape), dtype=torch.float32, device=device
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
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape), dtype=dtype, device=self._device
        )

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a batch from the buffer."""
        idx = torch.randint(
            0, self._size, (self._batch_size * self._num_envs,), device=self._device
        )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._batch_size)
        batch = {key: value[idx, env_idx] for key, value in self.data.items()}
        return batch
