# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
from typing import List, Optional

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


class QCritic(Critic):
    """Implementation of QCritic.

    A Q-function approximator that uses a multi-layer perceptron (MLP) to map observation-action pairs to Q-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own Q-function approximator by inheriting this class or :class:`Critic`.
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
        num_critics: int = 1,
        use_obs_encoder: bool = True,
        action_type: str = 'continuous',
    ) -> None:
        """Initialize the critic network.

        The Q critic network has two modes:

        -  ``use_obs_encoder`` = ``False`` :
           The input of the network is the concatenation of the observation and action.
        -  ``use_obs_encoder`` = ``True`` :
           The input of the network is the concatenation of the output of the observation encoder and action.

        For example, in :class:`DDPG`,
        the action is not directly concatenated with the observation,
        but is concatenated with the output of the observation encoder.

        .. note::
            The Q critic network contains multiple critics,
            and the output of the network :meth`forward` is a list of Q-values.
            If you want to get the single Q-value of a specific critic,
            you need to use the index to get it.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
            num_critics (int): Number of critics.
            use_obs_encoder (bool): Whether to use observation encoder.
        """
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
        expand_dim = act_dim if action_type == 'continuous' else 1
        for idx in range(num_critics):
            if self.use_obs_encoder:
                obs_encoder = build_mlp_network(
                    [obs_dim, hidden_sizes[0]],
                    activation=activation,
                    output_activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
                net = build_mlp_network(
                    [hidden_sizes[0] + expand_dim] + hidden_sizes[1:] + [1],
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
    ) -> List:
        """Forward function.

        As a multi-critic network, the output of the network is a list of Q-values.
        If you want to use it as a single-critic network,
        you only need to set the ``num_critics`` parameter to 1 when initializing the network,
        and then use the index 0 to get the Q-value.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
        """
        res = []
        for critic in self.critic_list:
            if self.use_obs_encoder:
                encodered_obs = critic[0](obs)
                res.append(torch.squeeze(critic[1](torch.cat([encodered_obs, act], dim=-1)), -1))
            else:
                res.append(torch.squeeze(critic[0](torch.cat([obs, act], dim=-1)), -1))
        return res
