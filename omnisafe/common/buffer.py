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
"""Implementation of the Buffer."""

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch

from omnisafe.utils.core import discount_cumsum
from omnisafe.utils.vtrace import calculate_v_trace


# pylint: disable-next=too-many-instance-attributes
class Buffer:
    """A buffer for storing trajectories experienced by an agent interacting with the environment.

    Besides, The buffer also provides the functionality of calculating the advantages of state-action pairs,
    ranging from ``GAE`` ,``V-trace`` to ``Plain`` method.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: tuple,
        act_dim: tuple,
        size: int,
        gamma: float,
        lam: float,
        adv_estimation_method: str,
        lam_c: float = 0.95,
        penalty_param: float = 0.0,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        r"""Initialize the buffer.

        .. note::

            .. list-table::

                *   -   obs_buf (np.array of shape (``batch_size``, ``obs_dim``)).
                    -   ``obsertaion`` in :meth:`roll_out` session.
                *   -   act_buf (np.array of shape (``batch_size``, ``act_dim``)).
                    -   ``action`` in :meth:`roll_out` session.
                *   -   adv_buf (np.array of shape (``batch_size``)
                    -   ``reward advantage`` calculated in :class:`Buffer`.
                *   -   discounted_ret_buf (np.array of shape (``batch_size``):
                    -   ``discounted return`` calculated in :class:`Buffer`.
                *   -   rew_buf (np.array of shape (``batch_size``):
                    -   ``reward`` in :meth:`roll_out` session.
                *   -   target_val_buf (np.array of shape (``batch_size``):
                    -   ``target value`` calculated in :class:`Buffer`,
                        which used to update reward critic in update session.
                *   -   val_buf (np.array of shape (``batch_size``):
                    -   ``value`` in :meth:`roll_out` session.
                *   -   logp_buf (np.array of shape (``batch_size``):
                    -   ``log probability`` in :meth:`roll_out` session.

        .. warning::
            Buffer collects only raw data received from environment.

        Args:
            obs_dim (tuple): The dimension of observation.
            act_dim (tuple): The dimension of action.
            size (int): The size of buffer.
            gamma (float): The discount factor.
            lam (float): The lambda factor for GAE.
            adv_estimation_method (str): The method for calculating advantages.
            lam_c (float, optional): The lambda factor for cost-based RL. Defaults to 0.95.
            reward_penalty (bool, optional): Whether to use reward penalty. Defaults to False.
        """
        self.size = size
        print(obs_dim)
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((size), dtype=np.float32)
        self.discounted_ret_buf = np.zeros((size), dtype=np.float32)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.target_val_buf = np.zeros((size), dtype=np.float32)
        self.val_buf = np.zeros((size), dtype=np.float32)
        self.logp_buf = np.zeros((size), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.lam_c = lam_c
        self.adv_estimation_method = adv_estimation_method
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

        # variables for cost-based RL
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(size, dtype=np.float32)
        self.target_cost_val_buf = np.zeros(size, dtype=np.float32)
        self.penalty_param = penalty_param
        self.device = device

        assert adv_estimation_method in ['gae', 'gae-rtg', 'vtrace', 'plain']

    def calculate_adv_and_value_targets(
        self,
        vals: np.ndarray,
        rews: np.ndarray,
        lam: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the estimated advantage.

        Three methods are supported:
        - GAE (Generalized Advantage Estimation)

        GAE is a variance reduction method for the actor-critic algorithm.
        It is proposed in the paper `High-Dimensional Continuous Control Using Generalized Advantage Estimation
        <https://arxiv.org/abs/1506.02438>`_.

        GAE calculates the advantage using the following formula:

        .. math::
            A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k}

        where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`
        When :math:`\lambda =1`, GAE reduces to the Monte Carlo method,
        which is unbiased but has high variance.
        When :math:`\lambda =0`, GAE reduces to the TD(1) method,
        which is biased but has low variance.

        - V-trace

        V-trace is a variance reduction method for the actor-critic algorithm.
        It is proposed in the paper
        `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
        <https://arxiv.org/abs/1802.01561>`_.

        V-trace calculates the advantage using the following formula:

        .. math::
            A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
            (\lambda \gamma)^n * \rho_{t+n} * (1 - d_{t+n}) * (V(x_{t+n}) - b_{t+n})

        where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`,
        :math:`\rho_{t+k} =\frac{\pi(a_{t+k}|s_{t+k})}{b_{t+k}}`,
        :math:`b_{t+k}` is the behavior policy,
        and :math:`d_{t+k}` is the done flag.

        - Plain

        Plain method is the original actor-critic algorithm.
        It is unbiased but has high variance.

        Args:
            vals (np.array): The value of states.
            rews (np.array): The reward of states.
            lam (float, optional): The lambda factor for GAE. Defaults to 0.95.
        """

        if self.adv_estimation_method == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv = discount_cumsum(deltas, self.gamma * lam)
            value_net_targets = adv + vals[:-1]

        elif self.adv_estimation_method == 'gae-rtg':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv = discount_cumsum(deltas, self.gamma * lam)
            # compute rewards-to-go, to be targets for the value function update
            value_net_targets = discount_cumsum(rews, self.gamma)[:-1]

        elif self.adv_estimation_method == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)
            log_p = self.logp_buf[path_slice]
            value_net_targets, adv, _ = calculate_v_trace(
                policy_action_probs=np.exp(log_p),
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

    # pylint: disable-next=too-many-arguments
    def store(
        self,
        obs: float,
        act: float,
        rew: float,
        val: float,
        logp: float,
        cost: float = 0.0,
        cost_val: float = 0.0,
    ) -> None:
        """Append one timestep of agent-environment interaction to the buffer.

        .. warning::
            - Store only raw data received from environment.
            - The buffer is not a circular queue.
              When the buffer is full, no more data will be stored.
        """
        assert self.ptr < self.max_size, 'No empty space in buffer'
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.cost_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.ptr += 1

    def finish_path(
        self,
        last_val: float = 0.0,
        last_cost_val: float = 0.0,
    ) -> None:
        """Call this at the end of a trajectory, or when one gets cut off by an epoch ending.

        This looks back in the buffer to where the trajectory started,
        and uses rewards and value estimates from the whole trajectory,
        to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state,
        to use as the targets for the value function.

        The ``last_val`` argument should be 0 if the trajectory ended,
        because the agent reached a ``terminal state`` (done),
        and otherwise should be :math:`V(s_T)`, the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account,
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).

        Args:
            last_val (float, optional): The value of last state. Defaults to 0.0.
            last_cost_val (float, optional): The cost value of last state. Defaults to 0.0.
            penalty_param (float, optional): The penalty parameter. Defaults to 0.0.
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        cost_vs = np.append(self.cost_val_buf[path_slice], last_cost_val)

        # new: add discounted returns to buffer
        discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
        self.discounted_ret_buf[path_slice] = discounted_ret
        assert self.penalty_param >= 0, 'reward_penalty assumes positive value.'
        rews -= self.penalty_param * costs

        adv, v_targets = self.calculate_adv_and_value_targets(vals, rews)
        self.adv_buf[path_slice] = adv
        self.target_val_buf[path_slice] = v_targets

        # calculate costs
        c_adv, c_targets = self.calculate_adv_and_value_targets(cost_vs, costs, lam=self.lam_c)
        self.cost_adv_buf[path_slice] = c_adv
        self.target_cost_val_buf[path_slice] = c_targets

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data from the buffer.

        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # if self.use_standardized_reward:
        # self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.mean() + 1.0e-8)

        # if self.use_standardized_cost:
        # also for cost advantages; only re-center but no rescale!
        # cadv_mean, *_ = distributed_utils.mpi_statistics_scalar(self.cost_adv_buf)
        # self.cost_adv_buf = self.cost_adv_buf - cadv_mean

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
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(self.size, dtype=np.float32)

        return {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in data.items()
        }

    def pre_process_data(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pre-process.

        .. note::
            For example, if ``use_standardized_obs`` is True,
            then the observations will be standardized,
            following the mean and std of the training data.
        """
        raw_data = self.get()
        data = deepcopy(raw_data)
        # Note: use_reward_scaling is currently applied in Buffer...
        # If self.use_reward_scaling:
        #     rew = self.ac.ret_oms(data['rew'], subtract_mean=False, clip=True)
        #     data['rew'] = rew
        return raw_data, data
