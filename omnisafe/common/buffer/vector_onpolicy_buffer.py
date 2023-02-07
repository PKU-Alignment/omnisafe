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
"""Implementation of vector on-policy buffer."""

from typing import Dict

import torch

from omnisafe.common.buffer import OnPolicyBuffer
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed_utils


class VectorOnPolicyBuffer(OnPolicyBuffer):
    """Vectorized on-policy buffer."""

    def __init__(  # pylint: disable=super-init-not-called, too-many-arguments
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
        device: torch.device = torch.device('cpu'),
    ):
        self._num_buffers = num_envs
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers = [
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
        """Get the number of buffers."""
        return self._num_buffers

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer."""
        for i, buffer in enumerate(self.buffers):
            buffer.store(**{k: v[i] for k, v in data.items()})

    def finish_path(
        self,
        last_val: torch.Tensor = torch.zeros(1),
        last_cost_val: torch.Tensor = torch.zeros(1),
        idx: int = 0,
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_val, last_cost_val)

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data from the buffer."""
        data_pre = {k: [v] for k, v in self.buffers[0].get().items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.get().items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}

        adv_mean, adv_std, *_ = distributed_utils.mpi_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed_utils.mpi_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data
