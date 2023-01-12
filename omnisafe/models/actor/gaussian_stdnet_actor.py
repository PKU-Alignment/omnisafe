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
"""Implementation of GaussianStdNetActor."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import build_mlp_network


class GaussianStdNetActor(Actor):
    """Implementation of GaussianStdNetActor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        activation,
        weight_initialization_mode,
        shared=None,
    ):
        """Initialize GaussianStdNetActor."""
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max

        if shared is not None:
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            std_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.mean = nn.Sequential(shared, mean_head)
            self.log_std = nn.Sequential(shared, std_head)
        else:
            net = build_mlp_network(
                [obs_dim] + list(hidden_sizes),
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            std_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.mean = nn.Sequential(net, mean_head)
            self.log_std = nn.Sequential(net, std_head)
        self.net = nn.ModuleList([self.mean, self.log_std])

    def _distribution(self, obs):
        mean = self.mean(obs)
        log_std = self.log_std(obs)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict(self, obs, deterministic=False, need_log_prob=False):
        dist = self._distribution(obs)
        if deterministic:
            out = dist.mean
        else:
            out = dist.rsample()

        action = torch.tanh(out)
        action = self.act_min + (action + 1) * 0.5 * (self.act_max - self.act_min)
        action = torch.clamp(action, self.act_min, self.act_max)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(out) ** 2).sum(axis=-1)
            return action.to(torch.float32), log_prob
        return action.to(torch.float32)

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        if act is not None:
            act = 2 * (act - self.act_min) / (self.act_max - self.act_min) - 1
            act = torch.tan(act)
            log_prob = dist.log_prob(act).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(act) ** 2).sum(axis=-1)
            return dist, log_prob
        return dist
