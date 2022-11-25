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

import torch
import torch.nn as nn

from omnisafe.algos.models.model_utils import build_mlp_network


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list, activation: str, shared=None) -> None:
        super().__init__()
        if shared is None:
            self.net = build_mlp_network([obs_dim] + hidden_sizes + [1], activation=activation)
        else:  # use shared layers
            value_head = nn.Linear(hidden_sizes[-1], 1)
            self.net = nn.Sequential(shared, value_head, nn.Identity())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Function Name.

        Args:
            obs: description

        Returns:
            description
        """
        return torch.squeeze(self.net(obs), -1)
