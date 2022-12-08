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
"""Implementation of ActorCritic."""

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

from omnisafe.models.actor import ActorBuilder
from omnisafe.models.critic import CriticBuilder
from omnisafe.utils.model_utils import build_mlp_network
from omnisafe.utils.online_mean_std import OnlineMeanStd


# pylint: disable=too-many-instance-attributes
class ActorCritic(nn.Module):
    """Actor-Critic methods"""

    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        scale_rewards: bool,
        model_cfgs,
    ) -> None:
        """Initialize ActorCritic"""
        super().__init__()

        self.obs_shape = observation_space.shape
        self.obs_dim = observation_space.shape[0]
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) if standardized_obs else None

        self.act_space_type = 'discrete' if isinstance(action_space, Discrete) else 'continuous'
        self.act_dim = action_space.shape[0] if isinstance(action_space, Box) else action_space.n

        self.model_cfgs = model_cfgs
        self.ac_kwargs = model_cfgs.ac_kwargs

        # Use for shared weights
        layer_units = [self.obs_dim] + self.ac_kwargs.pi.hidden_sizes
        activation = self.ac_kwargs.pi.activation
        if model_cfgs.shared_weights:
            self.shared = build_mlp_network(
                layer_units,
                activation=activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                output_activation=activation,
            )
        else:
            self.shared = None

        # Build actor
        actor_builder = ActorBuilder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.ac_kwargs.pi.hidden_sizes,
            activation=self.ac_kwargs.pi.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=self.shared,
        )
        if self.act_space_type == 'discrete':
            self.actor = actor_builder.build_actor('categorical')
        else:
            act_max = torch.as_tensor(action_space.high)
            act_min = torch.as_tensor(action_space.low)
            self.actor = actor_builder.build_actor(
                self.ac_kwargs.pi.actor_type, act_max=act_max, act_min=act_min
            )

        # Build critic
        critic_builder = CriticBuilder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.ac_kwargs.val.hidden_sizes,
            activation=self.ac_kwargs.val.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=self.shared,
        )
        self.reward_critic = critic_builder.build_critic('v')

        self.ret_oms = OnlineMeanStd(shape=(1,)) if scale_rewards else None

    def forward(self, obs):
        """Forward pass of the actor-critic model"""
        return self.step(obs)

    def step(self, obs, deterministic=False):
        """
        If training, this includes exploration noise!
        Expects that obs is not pre-processed.
        Args:
            obs, , description
        Returns:
            action, value, log_prob(action)
        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            value = self.reward_critic(obs)
            action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )

        return action.numpy(), value.numpy(), logp_a.numpy()

    def anneal_exploration(self, frac):
        """update internals of actors

        Updates exploration parameters for Gaussian actors update log_std

        Args:
            frac: progress of epochs, i.e. current epoch / total epochs
                    e.g. 10 / 100 = 0.1
        """
        if hasattr(self.actor, 'set_log_std'):
            self.actor.set_std(1 - frac)
