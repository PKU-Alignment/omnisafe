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
from torch import nn

from omnisafe.algos.models.actor import Actor
from omnisafe.algos.models.model_utils import build_mlp_network


class MLPActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_noise,
        hidden_sizes,
        act_limit,
        weight_initialization_mode,
        activation,
        shared=None,
    ):
        super().__init__(obs_dim, act_dim, weight_initialization_mode)
        self.act_limit = act_limit
        self.act_noise = act_noise
        if shared is not None:  # use shared layers
            action_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.net = nn.Sequential(shared, action_head, nn.Tanh())
        else:
            layers = [self.obs_dim] + hidden_sizes + [self.act_dim]
            self.net = build_mlp_network(
                layers,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
                output_activation='tanh',
            )

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.net(obs)

    def dist(self, obs):
        """
        Returns:
            torch.distributions.Distribution
        """
        pass

    def log_prob_from_dist(self, pi, act):
        """
        Returns:
            torch.Tensor
        """
        pass

    def predict(self, obs, determinstic=False):
        if determinstic == False:
            action = self.act_limit * self.net(obs)
            action += self.act_noise * np.random.randn(self.act_dim)
        else:
            action = self.act_limit * self.net(obs)
        return action.to(torch.float32), torch.tensor(1, dtype=torch.float32)
