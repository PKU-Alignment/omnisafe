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
"""Implementation of MLPActor."""

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class MLPActor(Actor):
    """A abstract class for actor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_noise,
        act_max,
        act_min,
        hidden_sizes: list,
        activation: Activation,
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation)
        self.act_max = act_max
        self.act_min = act_min
        self.act_noise = act_noise

        if shared is not None:  # use shared layers
            action_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation='tanh',
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, action_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim],
                activation=activation,
                output_activation='tanh',
                weight_initialization_mode=weight_initialization_mode,
            )

    def _distribution(self, obs):
        mean = self.net(obs)
        return Normal(mean, self._std)

    def forward(self, obs, act=None):
        """forward"""
        # Return output from network scaled to action space limits.
        return self.act_max * self.net(obs)

    def predict(self, obs, deterministic=False, need_log_prob=True):
        if deterministic:
            action = self.act_max * self.net(obs)
        else:
            action = self.act_max * self.net(obs)
            action += self.act_noise * np.random.randn(self.act_dim)

        action = torch.clamp(action, self.act_min, self.act_max)
        if need_log_prob:
            return action.to(torch.float32), torch.tensor(1, dtype=torch.float32)

        return action.to(torch.float32)
