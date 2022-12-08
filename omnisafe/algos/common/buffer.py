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
"""Buffer"""

import numpy as np
import torch

from omnisafe.algos.utils import distributed_utils
from omnisafe.algos.utils.core import combined_shape, discount_cumsum
from omnisafe.algos.utils.vtrace import calculate_v_trace


class Buffer:
    """Buffer API."""

    def __init__(
        self,
        actor_critic: torch.nn.Module,
        obs_dim: tuple,
        act_dim: tuple,
        size: int,
        gamma: float,
        lam: float,
        adv_estimation_method: str,
        scale_rewards: bool,
        standardized_obs: bool,
        standardized_reward: bool,
        standardized_cost: bool,
        lam_c: float = 0.95,
        reward_penalty: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        A buffer for storing trajectories experienced by an agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE)
        for calculating the advantages of state-action pairs.

        Important Note: Buffer collects only raw data received from environment.
        """
        self.actor_critic = actor_critic
        self.size = size
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.discounted_ret_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.target_val_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.lam_c = lam_c
        self.adv_estimation_method = adv_estimation_method
        self.use_scaled_rewards = scale_rewards
        self.use_standardized_obs = standardized_obs
        self.use_standardized_reward = standardized_reward
        self.use_standardized_cost = standardized_cost
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

        # variables for cost-based RL
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(size, dtype=np.float32)
        self.target_cost_val_buf = np.zeros(size, dtype=np.float32)
        self.use_reward_penalty = reward_penalty
        self.device = device

        assert adv_estimation_method in ['gae', 'gae2', 'vtrace', 'plain']

    def calculate_adv_and_value_targets(self, vals, rews, lam=None):
        """Compute the estimated advantage"""

        if self.adv_estimation_method == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv = discount_cumsum(deltas, self.gamma * lam)
            value_net_targets = adv + vals[:-1]

        elif self.adv_estimation_method == 'gae2':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            # print("deltas1",deltas)
            adv = discount_cumsum(deltas, self.gamma * lam)
            # print("adv1",adv)
            value_net_targets = discount_cumsum(rews, self.gamma)[:-1]

        elif self.adv_estimation_method == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)

            obs = (
                self.actor_critic.obs_oms(self.obs_buf[path_slice], clip=False)
                if self.standardize_env_obs
                else self.obs_buf[path_slice]
            )

            obs = torch.as_tensor(obs, dtype=torch.float32)

            act = self.act_buf[path_slice]
            act = torch.as_tensor(act, dtype=torch.float32)
            with torch.no_grad():
                # get current log_p of actions
                dist = self.actor_critic.pi.dist(obs)
                log_p = self.actor_critic.pi.log_prob_from_dist(dist, act)
            value_net_targets, adv, _ = calculate_v_trace(
                policy_action_probs=np.exp(log_p.numpy()),
                values=vals,
                rewards=rews,
                behavior_action_probs=np.exp(self.logp_buf[path_slice]),
                gamma=self.gamma,
                rho_bar=1.0,  # default is 1.0
                c_bar=1.0,  # default is 1.0
            )

        elif self.adv_estimation_method == 'plain':
            # A(x, u) = Q(x, u) - V(x) = r(x, u) + gamma V(x+1) - V(x)
            adv = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

            # compute rewards-to-go, to be targets for the value function update
            # value_net_targets are just the discounted returns
            value_net_targets = discount_cumsum(rews, self.gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, value_net_targets

    def store(self, obs, act, rew, val, logp, cost=0.0, cost_val=0.0):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Important Note: Store only raw data received from environment!!!
        Note: perform reward scaling if enabled
        """
        assert self.ptr < self.max_size, f'No empty space in buffer'

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.cost_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0, penalty_param=0):
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
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        cost_vs = np.append(self.cost_val_buf[path_slice], last_cost_val)

        # new: add discounted returns to buffer
        discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
        self.discounted_ret_buf[path_slice] = discounted_ret

        # if self.use_reward_penalty:
        #     assert penalty_param >= 0, 'reward_penalty assumes positive value.'
        #     rews -= penalty_param * costs

        # if self.use_scaled_rewards:
        #     # divide rewards by running return stddev.
        #     # discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
        #     # for i, ret in enumerate(discounted_ret):
        #     # update running return statistics
        #     # self.actor_critic.ret_oms.update(discounted_ret)
        #     # # now scale...
        #     rews = self.actor_critic.ret_oms(rews, subtract_mean=False, clip=True)

        adv, v_targets = self.calculate_adv_and_value_targets(vals, rews)
        self.adv_buf[path_slice] = adv
        self.target_val_buf[path_slice] = v_targets

        # calculate costs
        c_adv, c_targets = self.calculate_adv_and_value_targets(cost_vs, costs, lam=self.lam_c)
        self.cost_adv_buf[path_slice] = c_adv
        self.target_cost_val_buf[path_slice] = c_targets

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # TODO: pre-processing like standardization and scaling is done in
        #  Algorithm.  pre_process_data() method
        if self.use_standardized_reward:
            # the next two lines implement the advantage normalization trick
            # adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
            adv_mean, adv_std = distributed_utils.mpi_statistics_scalar(self.adv_buf)
            self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1.0e-8)
            # also for cost advantages; only re-center but no rescale!
            # cadv_mean, cadv_std = mpi_tools.mpi_statistics_scalar(self.cost_adv_buf)
            # self.cost_adv_buf = (self.cost_adv_buf - cadv_mean)#/(cadv_std + 1.0e-8)

        if self.use_standardized_cost:
            # also for cost advantages; only re-center but no rescale!
            cadv_mean, cadv_std = distributed_utils.mpi_statistics_scalar(self.cost_adv_buf)
            self.cost_adv_buf = (self.cost_adv_buf - cadv_mean) / (cadv_std + 1.0e-8)
        # TODO
        # self.obs_buf = self.actor_critic.obs_oms(self.obs_buf, clip=False) \
        #     if self.standardize_env_obs else self.obs_buf

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            target_v=self.target_val_buf,
            adv=self.adv_buf,
            log_p=self.logp_buf,
            discounted_ret=self.discounted_ret_buf,
            cost_adv=self.cost_adv_buf,
            target_c=self.target_cost_val_buf,
        )

        return {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in data.items()
        }
