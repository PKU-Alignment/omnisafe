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
from typing import Optional

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class QCritic(Critic):
    """Implementation of QCritic."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
        num_critics: int = 1,
        use_obs_encoder: bool = False,
    ) -> None:
        """Initialize."""
        self.use_obs_encoder = use_obs_encoder
        Critic.__init__(
            self,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.critic_list = []
        for idx in range(num_critics):
            if self.use_obs_encoder:
                obs_encoder = build_mlp_network(
                    [obs_dim, hidden_sizes[0]],
                    activation=activation,
                    output_activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
                net = build_mlp_network(
                    [hidden_sizes[0] + act_dim] + hidden_sizes[1:] + [1],
                    activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
                critic = nn.Sequential(obs_encoder, net)
            else:
                net = build_mlp_network(
                    [obs_dim + act_dim] + hidden_sizes[:] + [1],
                    activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
                critic = nn.Sequential(net)
            self.critic_list.append(critic)
            self.add_module(f'critic_{idx}', critic)

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
    ):
        """Forward."""
        res = []
        for critic in self.critic_list:
            if self.use_obs_encoder:
                encodered_obs = critic[0](obs)
                res.append(torch.squeeze(critic[1](torch.cat([encodered_obs, act], dim=-1)), -1))
            else:
                res.append(torch.squeeze(critic[0](torch.cat([obs, act], dim=-1)), -1))
        return res
