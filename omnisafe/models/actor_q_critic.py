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

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.critic.q_critic import QCritic
from omnisafe.utils.model_utils import build_mlp_network
from omnisafe.utils.online_mean_std import OnlineMeanStd

# pylint: disable-next=too-many-instance-attributes
class ActorQCritic(nn.Module):
    """Class for ActorQCritic."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        shared_weights: bool,
        model_cfgs,
        weight_initialization_mode='kaiming_uniform',
    ) -> None:
        """Initialize ActorQCritic"""
        super().__init__()

        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) if standardized_obs else None
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.ac_kwargs = model_cfgs.ac_kwargs
        # build policy and value functions
        if isinstance(action_space, Box):
            if model_cfgs.pi_type == 'dire':
                actor_fn = MLPActor
            act_dim = action_space.shape[0]
        else:
            raise ValueError

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

        self.actor = actor_fn(
            obs_dim=self.obs_dim,
            act_dim=act_dim,
            act_noise=model_cfgs.ac_kwargs.pi.act_noise,
            act_limit=self.act_limit,
            hidden_sizes=model_cfgs.ac_kwargs.pi.hidden_sizes,
            activation=model_cfgs.ac_kwargs.pi.activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.critic = QCritic(
            self.obs_dim,
            act_dim,
            hidden_sizes=model_cfgs.ac_kwargs.val.hidden_sizes,
            activation=model_cfgs.ac_kwargs.val.activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )
        self.critic_ = QCritic(
            self.obs_dim,
            act_dim,
            hidden_sizes=model_cfgs.ac_kwargs.val.hidden_sizes,
            activation=model_cfgs.ac_kwargs.val.activation,
            weight_initialization_mode=weight_initialization_mode,
            shared=shared,
        )

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
            if isinstance(self.pi, MLPActor):
                action = self.pi.predict(obs, determinstic=deterministic)
            else:
                action, logp_a = self.pi.predict(obs, determinstic=deterministic)
            value = self.v(obs, action)
            action = np.clip(action.numpy(), -self.act_limit, self.act_limit)

        return action, value.numpy(), logp_a.numpy()

    def anneal_exploration(self, frac):
        """update internals of actors
            1) Updates exploration parameters for Gaussian actors update log_std
        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1
        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)

    def forward(self, obs, act):
        """Compute the value of a given state-action pair."""
