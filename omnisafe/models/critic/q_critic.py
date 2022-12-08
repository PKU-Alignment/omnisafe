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
"""Implementation of QCritic."""

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class QCritic(Critic):
    """Implementation of QCritic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize."""
        Critic.__init__(
            self,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.obs_encoder = build_mlp_network(
            [obs_dim, hidden_sizes[0]],
            activation=activation,
            output_activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.net = build_mlp_network(
            [hidden_sizes[0] + act_dim] + hidden_sizes[1:] + [1],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """Forward."""
        obs = self.obs_encoder(obs)
        return torch.squeeze(self.net(torch.cat([obs, act], dim=-1)), -1)
