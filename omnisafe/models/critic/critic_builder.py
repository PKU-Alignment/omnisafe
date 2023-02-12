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
"""Implementation of CriticBuilder."""

from typing import Union

import torch.nn as nn

from omnisafe.models.critic.q_critic import QCritic
from omnisafe.models.critic.v_critic import VCritic
from omnisafe.typing import Activation, InitFunction


# pylint: disable-next=too-few-public-methods
class CriticBuilder:
    """Implementation of CriticBuilder

    .. note::

        A :class:`CriticBuilder` is a class for building a critic network.
        In ``omnisafe``, instead of building the critic network directly,
        we build it by integrating various types of critic networks into the :class:`CriticBuilder`.
        The advantage of this is that each type of critic has a uniform way of passing parameters.
        This makes it easy for users to use existing critics,
        and also facilitates the extension of new critic types.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize CriticBuilder.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weight_initialization_mode = weight_initialization_mode
        self.shared = shared

    def build_critic(
        self,
        critic_type: str,
        use_obs_encoder: bool = True,
    ) -> Union[QCritic, VCritic, NotImplementedError]:
        """Build critic.

        Currently, we support two types of critics: ``q`` and ``v``.
        If you want to add a new critic type, you can simply add it here.

        Args:
            critic_type (str): Critic type.
        """
        if critic_type == 'q':
            return QCritic(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
                use_obs_encoder=use_obs_encoder,
            )
        if critic_type == 'v':
            return VCritic(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
            )

        raise NotImplementedError(f'critic_type "{critic_type}" is not implemented.')
