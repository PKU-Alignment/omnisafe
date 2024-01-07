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
"""Implementation of OnPolicyBuffer."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum


class OnPolicyBuffer(BaseBuffer):  # pylint: disable=too-many-instance-attributes
    """A buffer for storing trajectories experienced by an agent interacting with the environment.

    Besides, The buffer also provides the functionality of calculating the advantages of
    state-action pairs, ranging from ``GAE``, ``GAE-RTG`` , ``V-trace`` to ``Plain`` method.

    .. warning::
        The buffer only supports ``Box`` and ``Discrete`` spaces.

    Compared to the base buffer, the on-policy buffer stores extra data:

    +----------------+---------+---------------+----------------------------------------+
    | Name           | Shape   | Dtype         | Shape                                  |
    +================+=========+===============+========================================+
    | discounted_ret | (size,) | torch.float32 | The discounted sum of return.          |
    +----------------+---------+---------------+----------------------------------------+
    | value_r        | (size,) | torch.float32 | The value estimated by reward critic.  |
    +----------------+---------+---------------+----------------------------------------+
    | value_c        | (size,) | torch.float32 | The value estimated by cost critic.    |
    +----------------+---------+---------------+----------------------------------------+
    | adv_r          | (size,) | torch.float32 | The advantage of the reward.           |
    +----------------+---------+---------------+----------------------------------------+
    | adv_c          | (size,) | torch.float32 | The advantage of the cost.             |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_r | (size,) | torch.float32 | The target value of the reward critic. |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_c | (size,) | torch.float32 | The target value of the cost critic.   |
    +----------------+---------+---------------+----------------------------------------+
    | logp           | (size,) | torch.float32 | The log probability of the action.     |
    +----------------+---------+---------------+----------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        gamma (float): The discount factor.
        lam (float): The lambda factor for calculating the advantages.
        lam_c (float): The lambda factor for calculating the advantages of the critic.
        advantage_estimator (AdvatageEstimator): The advantage estimator.
        penalty_coefficient (float, optional): The penalty coefficient. Defaults to 0.
        standardized_adv_r (bool, optional): Whether to standardize the advantages of the actor.
            Defaults to False.
        standardized_adv_c (bool, optional): Whether to standardize the advantages of the critic.
            Defaults to False.
        device (torch.device, optional): The device to store the data. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        ptr (int): The pointer of the buffer.
        path_start (int): The start index of the current path.
        max_size (int): The maximum size of the buffer.
        data (dict): The data stored in the buffer.
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        device (torch.device): The device to store the data.
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
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyBuffer`."""
        super().__init__(obs_space, act_space, size, device)

        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c
        self.data['adv_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)

        self._gamma: float = gamma
        self._lam: float = lam
        self._lam_c: float = lam_c
        self._penalty_coefficient: float = penalty_coefficient
        self._advantage_estimator: AdvatageEstimator = advantage_estimator
        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = size

        assert self._penalty_coefficient >= 0, 'penalty_coefficient must be non-negative!'
        assert self._advantage_estimator in ['gae', 'gae-rtg', 'vtrace', 'plain']

    @property
    def standardized_adv_r(self) -> bool:
        """Whether to standardize the advantages of the actor."""
        return self._standardized_adv_r

    @property
    def standardized_adv_c(self) -> bool:
        """Whether to standardize the advantages of the critic."""
        return self._standardized_adv_c

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer.

        .. warning::
            The total size of the data must be less than the buffer size.

        Args:
            data (torch.Tensor): The data to store.
        """
        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            self.data[key][self.ptr] = value
        self.ptr += 1

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs.

        On-policy algorithms need to calculate the advantages of state-action pairs
        after the path is finished. This function calculates the advantages of
        state-action pairs and stores them in the buffer, following the steps:

        .. hint::
            #. Calculate the discounted return.
            #. Calculate the advantages of the reward.
            #. Calculate the advantages of the cost.

        Args:
            last_value_r (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
            last_value_c (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        .. hint::
            We provide a trick to standardize the advantages of state-action pairs. We calculate the
            mean and standard deviation of the advantages of state-action pairs and then standardize
            the advantages of state-action pairs. You can turn on this trick by setting the
            ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
            cost.

        Returns:
            The data stored and calculated in the buffer.
        """
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

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the estimated advantage.

        Three methods are supported:

        - GAE (Generalized Advantage Estimation)

            GAE is a variance reduction method for the actor-critic algorithm. It is proposed in the
            paper `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_.

            GAE calculates the advantage using the following formula:

            .. math::

                A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k}

            where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`. When
            :math:`\lambda =1`, GAE reduces to the Monte Carlo method, which is unbiased but has high
            variance. When :math:`\lambda =0`, GAE reduces to the TD(1) method, which is biased but has
            low variance.

        - V-trace

            V-trace is a variance reduction method for the actor-critic algorithm. It is proposed in
            the paper `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures <https://arxiv.org/abs/1802.01561>`_.

            V-trace calculates the advantage using the following formula:

            .. math::

                A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
                    (\lambda \gamma)^n \rho_{t+n} (1 - d_{t+n}) (V(x_{t+n}) - b_{t+n})

            where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`,
            :math:`\rho_{t+k} = \frac{\pi(a_{t+k}|s_{t+k})}{b_{t+k}}`, :math:`b_{t+k}` is the
            behavior policy, and :math:`d_{t+k}` is the done flag.

        - Plain

            Plain method is the original actor-critic algorithm. It is unbiased but has high
            variance.

        Args:
            vals (torch.Tensor): The value of states.
            rews (torch.Tensor): The reward of states.
            lam (float): The lambda parameter in GAE formula.

        Returns:
            adv (torch.Tensor): The estimated advantage.
            target_value (torch.Tensor): The target value for the value function.

        Raises:
            NotImplementedError: If the advantage estimator is not supported.
        """  # pylint: disable=line-too-long
        if self._advantage_estimator == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            target_value = adv + values[:-1]

        elif self._advantage_estimator == 'gae-rtg':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            # compute rewards-to-go, to be targets for the value function update
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        elif self._advantage_estimator == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)
            action_probs = self.data['logp'][path_slice].exp()
            target_value, adv, _ = self._calculate_v_trace(
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
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, target_value

    @staticmethod
    # pylint: disable-next=too-many-arguments,too-many-locals
    def _calculate_v_trace(
        policy_action_probs: torch.Tensor,
        values: torch.Tensor,  # including bootstrap
        rewards: torch.Tensor,  # including bootstrap
        behavior_action_probs: torch.Tensor,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""This function is used to calculate V-trace targets.

        .. math::

            A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
                (\lambda \gamma)^n \rho_{t+n} (1 - d_{t+n}) (V(x_{t+n}) - b_{t+n})

        Calculate V-trace targets for off-policy actor-critic learning recursively. For more
        details, please refer to the paper: `Espeholt et al. 2018, IMPALA <https://arxiv.org/abs/1802.01561>`_.

        Args:
            policy_action_probs (torch.Tensor): Action probabilities of the policy.
            values (torch.Tensor): The value of states.
            rewards (torch.Tensor): The reward of states.
            behavior_action_probs (torch.Tensor): Action probabilities of the behavior policy.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            rho_bar (float, optional): The maximum value of importance weights. Defaults to 1.0.
            c_bar (float, optional): The maximum value of clipped importance weights. Defaults to 1.0.

        Returns:
            V-trace targets, shape=(batch_size, sequence_length)

        Raises:
            AssertionError: If the input tensors are scalars.
            AssertionError: If c_bar is greater than rho_bar.
        """
        assert values.ndim == 1, 'Please provide arrays instead of scalars'
        assert rewards.ndim == 1, 'Please provide arrays instead of scalars'
        assert policy_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert behavior_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert c_bar <= rho_bar, 'c_bar should be less than or equal to rho_bar'

        sequence_length = policy_action_probs.shape[0]
        # pylint: disable-next=assignment-from-no-return
        rhos = torch.div(policy_action_probs, behavior_action_probs)
        clip_rhos = torch.min(
            rhos,
            torch.as_tensor(rho_bar),
        )  # pylint: disable=assignment-from-no-return
        clip_cs = torch.min(
            rhos,
            torch.as_tensor(c_bar),
        )  # pylint: disable=assignment-from-no-return
        v_s = values[:-1].clone()  # copy all values except bootstrap value
        last_v_s = values[-1]  # bootstrap from last state

        # calculate v_s
        for index in reversed(range(sequence_length)):
            delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
            v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
            last_v_s = v_s[index]  # accumulate current v_s for next iteration

        # calculate q_targets
        v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
        policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

        return v_s, policy_advantage, clip_rhos
