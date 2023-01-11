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
"""Implementation of ActorQCritic."""

from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from omnisafe.models.actor import ActorBuilder
from omnisafe.models.critic.q_critic import QCritic
from omnisafe.utils.model_utils import build_mlp_network
from omnisafe.utils.online_mean_std import OnlineMeanStd


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
        shared_weights: bool,
        model_cfgs: NamedTuple,
        weight_initialization_mode='kaiming_uniform',
        device=torch.device('cpu'),
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
        self.act_dim = action_space.shape[0]
        self.act_max = torch.as_tensor(action_space.high).to(device)
        self.act_min = torch.as_tensor(action_space.low).to(device)
        self.ac_kwargs = model_cfgs.ac_kwargs
        # build policy and value functions

        self.obs_dim = observation_space.shape[0]

        # Use for shared weights
        layer_units = [self.obs_dim] + model_cfgs.ac_kwargs.pi.hidden_sizes

        activation = model_cfgs.ac_kwargs.pi.activation
        if shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
                output_activation=activation,
            )
        else:
            shared = None

        actor_builder = ActorBuilder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_noise=model_cfgs.ac_kwargs.pi.act_noise,
            hidden_sizes=model_cfgs.ac_kwargs.pi.hidden_sizes,
            activation=model_cfgs.ac_kwargs.pi.activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )

        if self.ac_kwargs.pi.actor_type == 'cholesky':
            self.actor = actor_builder.build_actor(
                self.ac_kwargs.pi.actor_type,
                act_max=self.act_max,
                act_min=self.act_min,
                cov_min=self.ac_kwargs.pi.cov_min,
                mu_clamp_min=self.ac_kwargs.pi.mu_clamp_min,
                mu_clamp_max=self.ac_kwargs.pi.mu_clamp_max,
                cov_clamp_min=self.ac_kwargs.pi.cov_clamp_min,
                cov_clamp_max=self.ac_kwargs.pi.cov_clamp_max,
            )
        else:
            self.actor = actor_builder.build_actor(
                self.ac_kwargs.pi.actor_type,
                act_max=self.act_max,
                act_min=self.act_min,
            )

        self.critic = QCritic(
            self.obs_dim,
            self.act_dim,
            hidden_sizes=model_cfgs.ac_kwargs.val.hidden_sizes,
            activation=model_cfgs.ac_kwargs.val.activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
            num_critics=model_cfgs.ac_kwargs.val.num_critics,
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
            action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )
            value = self.critic(obs, action)[0]
            action = action.to(torch.float32)

            return action.cpu().numpy(), value.cpu().numpy(), logp_a.cpu().numpy()

    def anneal_exploration(self, frac) -> None:
        """update internals of actors

        Updates exploration parameters for Gaussian actors update log_std

        Args:
            frac: progress of epochs. 1.0 is the end of training.
        """
        if hasattr(self.actor, 'set_std'):
            self.actor.set_log_std(1 - frac)
