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
"""Implementation of ActorBuilder."""

import difflib
from dataclasses import dataclass
from typing import Optional, Union

import torch.nn as nn

from omnisafe.models.actor.categorical_actor import CategoricalActor
from omnisafe.models.actor.cholesky_actor import MLPCholeskyActor
from omnisafe.models.actor.gaussian_actor import GaussianActor
from omnisafe.models.actor.gaussian_stdnet_actor import GaussianStdNetActor
from omnisafe.utils.model_utils import Activation, InitFunction


@dataclass
class NetworkConfig:
    """Class for storing network configurations."""

    obs_dim: int
    act_dim: int
    hidden_sizes: list
    activation: Activation = 'tanh'
    weight_initialization_mode: InitFunction = 'kaiming_uniform'
    shared: nn.Module = None
    output_activation: Optional[Activation] = None


@dataclass
class ActionConfig:
    """Class for storing action configurations."""

    scale_action: bool = False
    clip_action: bool = False
    std_learning: bool = True
    std_init: float = 1.0


# pylint: disable-next=too-few-public-methods
class ActorBuilder:
    """Class for building actor networks."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shared: nn.Module = None,
        scale_action: bool = False,
        clip_action: bool = False,
        output_activation: Optional[Activation] = 'identity',
        std_learning: bool = True,
        std_init: float = 1.0,
    ) -> None:
        """Initialize ActorBuilder."""
        self.network_config = NetworkConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.action_config = ActionConfig(
            scale_action=scale_action,
            clip_action=clip_action,
            std_learning=std_learning,
            std_init=std_init,
        )

    # pylint: disable-next=too-many-return-statements
    def build_actor(
        self, actor_type: str, **kwargs
    ) -> Union[
        CategoricalActor,
        GaussianStdNetActor,
        MLPCholeskyActor,
        GaussianActor,
        NotImplementedError,
    ]:
        """Build actor network."""
        if actor_type == 'categorical':
            return CategoricalActor(
                obs_dim=self.network_config.obs_dim,
                act_dim=self.network_config.act_dim,
                hidden_sizes=self.network_config.hidden_sizes,
                activation=self.network_config.activation,
                weight_initialization_mode=self.network_config.weight_initialization_mode,
                shared=self.network_config.shared,
                **kwargs,
            )
        if actor_type == 'gaussian_stdnet':
            return GaussianStdNetActor(
                obs_dim=self.network_config.obs_dim,
                act_dim=self.network_config.act_dim,
                hidden_sizes=self.network_config.hidden_sizes,
                activation=self.network_config.activation,
                weight_initialization_mode=self.network_config.weight_initialization_mode,
                shared=self.network_config.shared,
                scale_action=self.action_config.scale_action,
                **kwargs,
            )
        if actor_type == 'cholesky':
            return MLPCholeskyActor(
                obs_dim=self.network_config.obs_dim,
                act_dim=self.network_config.act_dim,
                hidden_sizes=self.network_config.hidden_sizes,
                activation=self.network_config.activation,
                weight_initialization_mode=self.network_config.weight_initialization_mode,
                **kwargs,
            )
        if actor_type == 'gaussian':
            return GaussianActor(
                obs_dim=self.network_config.obs_dim,
                act_dim=self.network_config.act_dim,
                hidden_sizes=self.network_config.hidden_sizes,
                activation=self.network_config.activation,
                weight_initialization_mode=self.network_config.weight_initialization_mode,
                scale_action=self.action_config.scale_action,
                clip_action=self.action_config.clip_action,
                output_activation=self.network_config.output_activation,
                std_learning=self.action_config.std_learning,
                std_init=self.action_config.std_init,
                shared=self.network_config.shared,
                **kwargs,
            )

        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! Did you mean \
                {difflib.get_close_matches(actor_type, ["categorical", "gaussian_stdnet", "cholesky", "gaussian"], n=1)[0]}?'  # pylint: disable-next=line-too-long
        )
