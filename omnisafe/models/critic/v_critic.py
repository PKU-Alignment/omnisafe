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
"""Implementation of VCritic."""

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class VCritic(Critic):
    """Implementation of VCritic.

    A V-function approximator that uses a multi-layer perceptron (MLP) to map observations to V-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own V-function approximator by inheriting this class or :class:`Critic`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize the critic network.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        Critic.__init__(
            self,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        if shared is not None:
            value_head = build_mlp_network(
                sizes=[hidden_sizes[-1], 1],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, value_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [1],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations.
            act (torch.Tensor): Actions.
        """
        return torch.squeeze(self.net(obs), -1)
