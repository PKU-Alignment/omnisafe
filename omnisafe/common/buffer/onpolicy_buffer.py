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
"""Implementation of on-policy buffer."""

from typing import Dict, Tuple

import torch

from omnisafe.common.buffer import BaseBuffer
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed_utils
from omnisafe.utils.core import discount_cumsum_torch
from omnisafe.utils.vtrace import calculate_v_trace


class OnPolicyBuffer(BaseBuffer):  # pylint: disable=too-many-instance-attributes
    """A buffer for storing trajectories experienced by an agent interacting with the environment.

    Besides, The buffer also provides the functionality of calculating the advantages of state-action pairs,
    ranging from ``GAE`` ,``V-trace`` to ``Plain`` method.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(obs_space, act_space, size, device)
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        self.data['adv_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)

        self._gamma = gamma
        self._lam = lam
        self._lam_c = lam_c
        self._penalty_coefficient = penalty_coefficient
        self._advantage_estimator = advantage_estimator

        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size = size

        assert self._penalty_coefficient >= 0, 'penalty_coefficient must be non-negative!'
        assert self._advantage_estimator in ['gae', 'gae-rtg', 'vtrace', 'plain']

    @property
    def standardized_adv_r(self) -> bool:
        """Get the standardized_adv_r."""
        return self._standardized_adv_r

    @property
    def standardized_adv_c(self) -> bool:
        """Get the standardized_adv_c."""
        return self._standardized_adv_c

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer."""
        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            self.data[key][self.ptr] = value
        self.ptr += 1

    def finish_path(
        self,
        last_val: torch.Tensor = torch.zeros(1),
        last_cost_val: torch.Tensor = torch.zeros(1),
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr)
        last_val = last_val.to(self.device)
        last_cost_val = last_cost_val.to(self.device)
        rewards = torch.cat([self.data['reward'][path_slice], last_val])
        values_r = torch.cat([self.data['value_r'][path_slice], last_val])
        costs = torch.cat([self.data['cost'][path_slice], last_cost_val])
        values_c = torch.cat([self.data['value_c'][path_slice], last_cost_val])

        discountred_ret = discount_cumsum_torch(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            rewards, values_r, lam=self._lam
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            costs, values_c, lam=self._lam_c
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer."""
        assert self.ptr == self.max_size, 'The buffer is not full!'
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
        }

        self.data['adv_r'] = torch.zeros_like(self.data['adv_r'])
        self.data['adv_c'] = torch.zeros_like(self.data['adv_c'])

        adv_mean, adv_std, *_ = distributed_utils.mpi_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed_utils.mpi_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data

    def _calculate_adv_and_value_targets(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if self._advantage_estimator == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum_torch(deltas, self._gamma * lam)
            target_value = adv + values[:-1]

        elif self._advantage_estimator == 'gae-rtg':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum_torch(deltas, self._gamma * lam)
            # compute rewards-to-go, to be targets for the value function update
            target_value = discount_cumsum_torch(rewards, self._gamma)[:-1]

        elif self._advantage_estimator == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)
            action_probs = self.data['logp'][path_slice].exp()
            target_value, adv, _ = calculate_v_trace(
                policy_action_probs=action_probs,
                values=values,
                rewards=rewards,
                behavior_action_probs=action_probs,
                gamma=self._gamma,
                rho_bar=1.0,
                c_bar=1.0,
            )

        elif self._advantage_estimator == 'plain':
            # A(x, u) = Q(x, u) - V(x) = r(x, u) + gamma V(x+1) - V(x)
            adv = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            target_value = discount_cumsum_torch(rewards, self._gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, target_value
