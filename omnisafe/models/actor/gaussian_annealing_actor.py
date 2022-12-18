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
"""Implementation of GaussianAnnealingActor."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import build_mlp_network


class GaussianAnnealingActor(Actor):
    """Class for Gaussian Annealing Actor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_min: torch.Tensor,
        act_max: torch.Tensor,
        hidden_sizes,
        activation,
        weight_initialization_mode,
        shared=None,
        start_std: float = 0.5,
        end_std: float = 0.01,
    ):
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.start_std = start_std
        self.end_std = end_std
        self._std = self.start_std * torch.ones(self.act_dim, dtype=torch.float32)

        self.act_min = act_min
        self.act_max = act_max

        if shared is not None:
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

    def predict(self, obs, deterministic=False, need_log_prob=False):
        dist = self._distribution(obs)
        if deterministic:
            out = dist.mean
        else:
            out = dist.sample()

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            return out, log_prob
        return out

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        if act is not None:
            act = 2 * (act - self.act_min) / (self.act_max - self.act_min) - 1
            log_prob = dist.log_prob(act).sum(axis=-1)
            return dist, log_prob
        return dist

    def set_std(self, proportion):
        """To support annealing exploration noise.
        proportion is annealing from 1. to 0 over course of training"""
        std = self.start_std * proportion + self.end_std * (1 - proportion)
        self._std = std * torch.ones(self.act_dim, dtype=torch.float32)

    @property
    def std(self):
        """Return the current std of the Gaussian distribution."""
        return self._std.mean().item()
