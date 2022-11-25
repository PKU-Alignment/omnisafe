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

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

from omnisafe.algos.models.critic import Critic
from omnisafe.algos.models.mlp_categorical_actor import MLPCategoricalActor
from omnisafe.algos.models.mlp_gaussian_actor import MLPGaussianActor
from omnisafe.algos.models.model_utils import build_mlp_network
from omnisafe.algos.models.online_mean_std import OnlineMeanStd


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        scale_rewards: bool,
        shared_weights: bool,
        ac_kwargs: dict,
        weight_initialization_mode='kaiming_uniform',
    ) -> None:
        super().__init__()

        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) if standardized_obs else None

        self.ac_kwargs = ac_kwargs

        # policy builder depends on action space
        if isinstance(action_space, Box):
            actor_fn = MLPGaussianActor
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            actor_fn = MLPCategoricalActor
            act_dim = action_space.n
        else:
            raise ValueError

        obs_dim = observation_space.shape[0]

        # Use for shared weights
        layer_units = [obs_dim] + ac_kwargs['pi']['hidden_sizes']

        activation = ac_kwargs['pi']['activation']
        if shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
                output_activation=activation,
            )
        else:
            shared = None

        self.pi = actor_fn(
            obs_dim=obs_dim,
            act_dim=act_dim,
            shared=shared,
            weight_initialization_mode=weight_initialization_mode,
            **ac_kwargs['pi'],
        )
        self.v = Critic(obs_dim, shared=shared, **ac_kwargs['val'])

        self.ret_oms = OnlineMeanStd(shape=(1,)) if scale_rewards else None

    def forward(self, obs):
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
            v = self.v(obs)
            a, logp_a = self.pi.predict(obs, deterministic=deterministic)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def anneal_exploration(self, frac):
        """update internals of actors
            1) Updates exploration parameters for Gaussian actors update log_std
        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1
        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)
