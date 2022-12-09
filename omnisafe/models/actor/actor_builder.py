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

import torch.nn as nn

from omnisafe.models.actor.categorical_actor import CategoricalActor
from omnisafe.models.actor.gaussian_annealing_actor import GaussianAnnealingActor
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_stdnet_actor import GaussianStdNetActor
from omnisafe.utils.model_utils import Activation, InitFunction


class ActorBuilder:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weight_initialization_mode = weight_initialization_mode
        self.shared = shared

    def build_actor(self, actor_type: str, **kwargs):
        if actor_type == 'categorical':
            return CategoricalActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
                **kwargs,
            )
        elif actor_type == 'gaussian_annealing':
            return GaussianAnnealingActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
                **kwargs,
            )
        elif actor_type == 'gaussian_stdnet':
            return GaussianStdNetActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
                **kwargs,
            )
        elif actor_type == 'gaussian_learning':
            return GaussianLearningActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
                **kwargs,
            )
        else:
            raise NotImplementedError(f'Actor type {actor_type} is not implemented.')
