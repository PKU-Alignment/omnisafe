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

from omnisafe.models.actor.categorical_actor import CategoricalActor
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.actor.perturbation_actor import PerturbationActor
from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, ActorType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class ActorBuilder:
    """Class for building actor networks.

    Actor networks are used in the Actor design of Reinforcement Learning (RL)
    to choose actions based on the current state of the environment.

    Attributes:
        obs_space (OmnisafeSpace): The space that defines valid observations.
        act_space (OmnisafeSpace): The space that defines valid actions.
        hidden_sizes (list[int]): The number of nodes at each hidden layer in the network.
        activation (str, optional): The activation function used after each layer. Defaults to ``'relu'``.
        weight_initialization_mode (str, optional): The method to initialize weights in the network.
                                                    Defaults to ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`ActorBuilder`."""
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes

    # pylint: disable-next=too-many-return-statements
    def build_actor(
        self,
        actor_type: ActorType,
    ) -> Actor:
        """Generate an actor model of the given type using preset parameters.

        The supported actor types include:
            - `gaussian_learning`: Gaussian actor with learnable standard deviation parameters.
            - `gaussian_sac`: Gaussian actor with learnable standard deviation network.
            - `mlp`: Multi-layer perceptron actor, typically used in DDPG and TD3.
            - `vae`: Variational AutoEncoder actor, used for continual and low-data learning.
            - `perturbation`: Perturbation Actor for domain randomization.
            - `discrete`: Discrete/Categorical actor, used in environments with discrete action spaces.

        Args:
            actor_type (str): The type of actor network to build.

        Returns:
            Actor: An instance of the requested actor model.

        Raises:
            NotImplementedError: If the requested actor type has not been implemented.
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
        if actor_type == 'vae':
            return VAE(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'perturbation':
            return PerturbationActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'discrete':
            return CategoricalActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! '
            f'Available actor types are: gaussian_learning, gaussian_sac, mlp, vae, perturbation, discrete.',
        )
