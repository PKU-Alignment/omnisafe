# Copyright 2023 OmniSafe Team. All Rights Reserved.
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

from __future__ import annotations

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, ActorType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class ActorBuilder:
    """Class for building actor networks.

    Attributes:
        _obs_space (OmnisafeSpace): Observation space.
        _act_space (OmnisafeSpace): Action space.
        _weight_initialization_mode (InitFunction): Weight initialization mode.
        _activation (Activation): Activation function.
        _hidden_sizes (list[int]): List of hidden layer sizes.

    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize ActorBuilder.

        Args:
            obs_space (OmnisafeSpace): Observation space.
            act_space (OmnisafeSpace): Action space.
            hidden_sizes (list): List of hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
        """
        self._obs_space = obs_space
        self._act_space = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes = hidden_sizes

    # pylint: disable-next=too-many-return-statements
    def build_actor(self, actor_type: ActorType) -> Actor:
        """Build actor network.

        Currently, we support the following actor types:
            - ``gaussian_learning``: Gaussian actor with learnable standard deviation parameters.
            - ``gaussian_sac``: Gaussian actor with learnable standard deviation network.
            - ``mlp``: Multi-layer perceptron actor, used in ``DDPG`` and ``TD3``.

        Args:
            actor_type (ActorType): Actor type.
        """
        if actor_type == 'gaussian_learning':
            return GaussianLearningActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'gaussian_sac':
            return GaussianSACActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'mlp':
            return MLPActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! '
            f'Available actor types are: gaussian_learning, gaussian_sac, mlp.',
        )
