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
"""Implementation of ActorQCritic."""

from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

from omnisafe.models.actor import ActorBuilder
from omnisafe.models.critic.q_critic import QCritic
from omnisafe.utils.model_utils import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class ActorQCritic(nn.Module):
    """Class for ActorCritic.

    In ``omnisafe``, we combine the actor and critic into one this class.

    .. list-table::

        *   -   Model
            -   Description
            -   Function
        *   -   Actor
            -   The policy network, input is observation, output is action.
                Choose the actor from the following options:
                :class:`MLPActor`, :class:`CategoricalActor`, :class:`GaussianAnnealingActor`,
                :class:`GaussianLearningActor`, :class:`GaussianStdNetActor`, :class:`MLPCholeskyActor`.
            -   Choose the action based on the observation.
        *   -   Value Q Critic
            -   The value network, input is observation-action pair,
                output is reward value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the reward value of the observation.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        model_cfgs: NamedTuple,
    ) -> None:
        """Initialize ActorQCritic

        .. note::
            Instead of creating the actor or critic directly, we use the builder to create them.
            The advantage of this is that,
            each type of critic has a uniform way of passing parameters.
            This makes it easy for users to use existing critics,
            and also facilitates the extension of new critic types.

        Args:
            observation_space: observation space
            action_space: action space
            standardized_obs: whether to standardize observation
            shared_weights: whether to share weights between actor and critic
            model_cfgs: model configurations
            weight_initialization_mode: weight initialization mode
            device: device, cpu or cuda
        """
        super().__init__()

        self.obs_shape = observation_space.shape
        self.act_dim = action_space.shape[-1] if isinstance(action_space, Box) else action_space.n
        self.ac_kwargs = model_cfgs.ac_kwargs
        # build policy and value functions
        self.act_space_type = 'discrete' if isinstance(action_space, Discrete) else 'continuous'
        self.obs_dim = observation_space.shape[0]

        # Use for shared weights
        layer_units = [self.obs_dim] + model_cfgs.ac_kwargs.pi.hidden_sizes

        activation = model_cfgs.ac_kwargs.pi.activation
        if model_cfgs.shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                output_activation=activation,
            )
        else:
            shared = None
        actor_builder = ActorBuilder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=shared,
            **model_cfgs.ac_kwargs.pi,
        )

        if model_cfgs.actor_type == 'cholesky':
            self.actor = actor_builder.build_actor(
                model_cfgs.actor_type,
                act_max=torch.as_tensor(action_space.high),
                act_min=torch.as_tensor(action_space.low),
                cov_min=model_cfgs.cov_min,
                mu_clamp_min=model_cfgs.mu_clamp_min,
                mu_clamp_max=model_cfgs.mu_clamp_max,
                cov_clamp_min=model_cfgs.cov_clamp_min,
                cov_clamp_max=model_cfgs.cov_clamp_max,
            )
        elif self.act_space_type == 'discrete':
            self.actor = actor_builder.build_actor('categorical')
        else:
            act_max = torch.as_tensor(action_space.high)
            act_min = torch.as_tensor(action_space.low)
            self.actor = actor_builder.build_actor(
                model_cfgs.actor_type,
                act_max=act_max,
                act_min=act_min,
            )

        self.critic = QCritic(
            self.obs_dim,
            self.act_dim,
            hidden_sizes=model_cfgs.ac_kwargs.val.hidden_sizes,
            activation=model_cfgs.ac_kwargs.val.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=shared,
            num_critics=model_cfgs.ac_kwargs.val.num_critics,
            action_type='continuous' if isinstance(action_space, Box) else 'discrete',
        )

    def forward(self, obs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass of the actor-critic model"""
        return self.step(obs)

    def step(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step function of the actor-critic model

        Input observation, output value (from :class:`Critic`) action,
        and its log probability (from :class`Actor`).

        .. note::
            The observation is standardized by the running mean and standard deviation.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to use deterministic action.
        """
        with torch.no_grad():
            raw_action, action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )
            value = self.critic(obs, action)[0]

        return raw_action, action, value, logp_a

    def anneal_exploration(self, frac: float) -> None:
        """Update internals of actors

        Updates exploration parameters for Gaussian actors update log_std

        Args:
            frac: progress of epochs. 1.0 is the end of training.
        """
        if hasattr(self.actor, 'set_std'):
            self.actor.set_std(1 - frac)
