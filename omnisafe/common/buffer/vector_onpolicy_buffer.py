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
"""Implementation of VectorOnPolicyBuffer."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed


class VectorOnPolicyBuffer(OnPolicyBuffer):
    """Vectorized on-policy buffer.

    The vector-on-policy buffer is used to store the data from vector environments. The data is
    stored in a list of on-policy buffers, each of which corresponds to one environment.

    .. warning::
        The buffer only supports ``Box`` and ``Discrete`` spaces.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        size (int): Size of the buffer.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.
        lam_c (float): Lambda for GAE for cost.
        advantage_estimator (AdvatageEstimator): Advantage estimator.
        penalty_coefficient (float): Penalty coefficient.
        standardized_adv_r (bool): Whether to standardize the advantage for reward.
        standardized_adv_c (bool): Whether to standardize the advantage for cost.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        buffers (list[OnPolicyBuffer]): List of on-policy buffers.
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOnPolicyBuffer`."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c

        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers: list[OnPolicyBuffer] = [
            OnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    @property
    def num_buffers(self) -> int:
        """Number of buffers."""
        return self._num_buffers

    def store(self, **data: torch.Tensor) -> None:
        """Store vectorized data into vectorized buffer."""
        for i, buffer in enumerate(self.buffers):
            buffer.store(**{k: v[i] for k, v in data.items()})

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """Get the data in the buffer.

        In vector-on-policy buffer, we get the data from each buffer and then concatenate them.
        """
        self.buffers[idx].finish_path(last_value_r, last_value_c)

    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        We provide a trick to standardize the advantages of state-action pairs. We calculate the
        mean and standard deviation of the advantages of state-action pairs and then standardize
        the advantages of state-action pairs. You can turn on this trick by setting the
        ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
        cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        data_pre = {k: [v] for k, v in self.buffers[0].get().items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.get().items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data
