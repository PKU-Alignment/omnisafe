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

import random
from operator import itemgetter

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, cost, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[
                : len(self.buffer) - self.position
            ]
            self.buffer[: len(batch) - len(self.buffer) + self.position] = batch[
                len(self.buffer) - self.position :
            ]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def return_all(self):
        state, action, reward, cost, next_state, done = map(np.stack, zip(*self.buffer))
        return state, action, reward, cost, next_state, done

    def __len__(self):
        return len(self.buffer)


#  Code from spinning-up repository: https://github.com/openai/spinningup


# import core as core
import omnisafe.algorithms.model_based.models.core_sac as core_sac


class SAC_ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size=int(1e6), device=torch.device('cpu')):
        self.state = np.zeros(core_sac.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(core_sac.combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(core_sac.combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.cost = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = torch.device(device)

    def store(self, obs, act, rew, next_obs, done, cost=None):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        if cost is not None:
            self.cost[self.ptr] = cost
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.state[idxs],
            obs2=self.next_state[idxs],
            act=self.action[idxs],
            rew=self.reward[idxs],
            cost=self.cost[idxs],
            done=self.done[idxs],
        )
        return_dict = {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()
        }
        return_dict['idx'] = idxs
        return return_dict


from omnisafe.algorithms.model_based.models import core_ac
from omnisafe.utils.distributed_utils import mpi_statistics_scalar


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97, device=torch.device('cpu')):
        self.obs_buf = np.zeros(core_ac.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core_ac.combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.device = torch.device(device)

    def store(self, obs, act, rew, crew, val, cval, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core_ac.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core_ac.discount_cumsum(cdeltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core_ac.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core_ac.discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)
        # print(adv_mean, adv_std, cadv_mean, cadv_std)
        eps = 1e-5
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = self.cadv_buf - cadv_mean  # / (cadv_std+eps)
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            cret=self.cret_buf,
            adv=self.adv_buf,
            cadv=self.cadv_buf,
            logp=self.logp_buf,
        )
        return {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in data.items()
        }
