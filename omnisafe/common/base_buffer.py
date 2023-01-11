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

import numpy as np
import torch

from omnisafe.typing import Dict
from omnisafe.utils.core import combined_shape


# pylint: disable-next=too-many-instance-attributes, too-many-arguments
class BaseBuffer:
    """A simple FIFO (First In First Out) experience replay buffer for off-policy agents."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: tuple,
        act_dim: tuple,
        size: int,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ):
        r"""Initialize the replay buffer.

        .. note::

            .. list-table::

                *   -   obs_buf (np.ndarray of shape ``(batch_size, obs_dim)``).
                    -   ``obsertaion`` in :meth:`roll_out` session.
                *   -   act_buf (np.ndarray of shape ``(batch_size, act_dim)``).
                    -   ``action`` in :meth:`roll_out` session.
                *   -   rew_buf (np.ndarray of shape ``batch_size``).
                    -   ``reward`` in :meth:`roll_out` session.
                *   -   cost_buf (np.ndarray of shape shape ``batch_size``).
                    -   ``cost`` in :meth:`roll_out` session.
                *   -   next_obs_buf (np.ndarray of shape ``(batch_size, obs_dim)``).
                    -   ``next obsertaion`` in :meth:`roll_out` session.
                *   -   done_buf (np.ndarray of shape shape ``batch_size``).
                    -   ``terminated`` in :meth:`roll_out` session.

        Args:
            obs_dim (int): observation dimension.
            act_dim (int): action dimension.
            size (int): buffer size.
            batch_size (int): batch size.
        """
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.pre_cost_buf = np.zeros(size, dtype=np.float32)  # previous cost for safety layer
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size
        self.device = device

    # pylint: disable-next=too-many-arguments
    def store(
        self,
        obs: float,
        act: float,
        rew: float,
        cost: float,
        next_obs: float,
        done: float,
        pre_cost: float = 0.0,
    ) -> None:
        """Store the experience in the buffer.

        .. note::
            Replay buffer stores the experience following the rule of FIFO (First In First Out).
            Besides, The buffer is a circular queue, which means that when the buffer is full,
            the oldest experience will be replaced by the newest experience.

        Args:
            obs (float): observation.
            act (float): action.
            rew (float): reward.
            cost (float): cost.
            next_obs (float): next observation.
            done (float): terminated.
        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.pre_cost_buf[self.ptr] = pre_cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences from the buffer."""
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            cost=self.cost_buf[idxs],
            done=self.done_buf[idxs],
            pre_cost=self.pre_cost_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class VectorBaseBuffer:
    """A simple FIFO (First In First Out) experience replay buffer for off-policy agents."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: tuple,
        act_dim: tuple,
        size: int,
        batch_size: int,
        num_envs: int,
    ):
        """Initialize the replay buffer."""
        self.num_envs = num_envs
        self.buffers = [BaseBuffer(obs_dim, act_dim, size, batch_size) for _ in range(num_envs)]

    # pylint: disable-next=too-many-arguments
    def store(self, obs, act, rew, cost, next_obs, done, pre_cost=0.0):
        """Store the experience in the buffer."""
        for i in range(self.num_envs):
            self.buffers[i].store(obs[i], act[i], rew[i], cost[i], next_obs[i], done[i], pre_cost)

    def sample_batch(self):
        """Sample a batch of experiences from the buffer."""
        data = {}
        for _, buffer in enumerate(self.buffers):
            buffer_data = buffer.sample_batch()
            for key, value in buffer_data.items():
                if key in data:
                    data[key] = torch.cat((data[key], value), dim=0)
                else:
                    data[key] = value
        return data
