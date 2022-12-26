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

import torch
import torch.nn as nn

from omnisafe.models.actor import ActorBuilder
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
        device=torch.device('cpu'),
    ) -> None:
        """Initialize ActorQCritic"""
        super().__init__()

        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) if standardized_obs else None
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
            action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )
            value = self.critic(obs, action)[0]
            action = action.to(torch.float32)

            return action.cpu().numpy(), value.cpu().numpy(), logp_a.cpu().numpy()

    def anneal_exploration(self, frac):
        """update internals of actors
            1) Updates exploration parameters for Gaussian actors update log_std
        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1
        """
        if hasattr(self.actor, 'set_log_std'):
            self.actor.set_log_std(1 - frac)

    def forward(self, obs, act):
        """Compute the value of a given state-action pair."""
