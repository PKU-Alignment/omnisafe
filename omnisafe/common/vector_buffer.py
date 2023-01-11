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

from omnisafe.utils import distributed_utils
from omnisafe.utils.core import combined_shape, discount_cumsum
from omnisafe.utils.vtrace import calculate_v_trace
from omnisafe.common.buffer import Buffer

class VectorBuffer:

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: tuple,
        act_dim: tuple,
        size: int,
        gamma: float,
        lam: float,
        adv_estimation_method: str,
        standardized_rew_adv: bool,
        standardized_cost_adv: bool,
        lam_c: float = 0.95,
        penalty_param: float = 0.0,
        device: torch.device = torch.device('cpu'),
        num_envs: int = 1,
    ) -> None:
        """Initialize the buffer."""
        self.num_buffer = num_envs
        self.standardized_rew_adv = standardized_rew_adv
        self.standardized_cost_adv = standardized_cost_adv
        if num_envs < 1:
            raise ValueError("num_envs must be greater than 0.")
        self.buffers = []
        for _ in range(num_envs):
            self.buffers.append(
                Buffer(
                    obs_dim,
                    act_dim,
                    size,
                    gamma,
                    lam,
                    adv_estimation_method,
                    lam_c,
                    penalty_param,
                    device,
                )
            )
    
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
        """Store one step of interaction."""
        for i, buffer in enumerate(self.buffers):
            buffer.store(obs[i], act[i], rew[i], val[i], logp[i], cost[i], cost_val[i])

    def finish_path(
        self, 
        last_val: float, 
        last_cost_val: float, 
        idx: int = 0,
        ) -> None:
        """Finish one trajectory."""
        self.buffers[idx].finish_path(last_val, last_cost_val)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data in buffers."""
        data = {}
        for i, buffer in enumerate(self.buffers):
            buffer_data = buffer.get()
            for key, value in buffer_data.items():
                if key in data:
                    data[key] = torch.cat((data[key], value), dim=0)
                else:
                    data[key] = value
        adv_mean, adv_std, *_ = distributed_utils.mpi_statistics_scalar(data['adv'])
        cadv_mean, cadv_std, *_ = distributed_utils.mpi_statistics_scalar(data['cost_adv'])
        if self.standardized_rew_adv:
            data['adv'] = (data['adv'] - adv_mean)/(adv_std+1e-8)
        if self.standardized_cost_adv:
            data['cost_adv'] = data['cost_adv'] - cadv_mean
        return data
    


