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
from omnisafe.utils.model_utils import Activation, build_mlp_network


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
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: Activation = 'kaiming_uniform',
        shared=None,
        scale_action=False,
        clip_action: bool = False,
    ):
        """Initialize GaussianStdNetActor."""
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max
        self.scale_action = scale_action
        self.clip_action = clip_action

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
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            std_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation=output_activation,
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

        if self.scale_action:
            self.act_min = self.act_min.to(out.device)
            self.act_max = self.act_max.to(out.device)
            action = self.act_min + (out + 1) / 2 * (self.act_max - self.act_min)
        else:
            action = out

        if self.clip_action:
            action = torch.clamp(action, self.act_min, self.act_max)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(out) ** 2).sum(axis=-1)
            return out.to(torch.float32), action.to(torch.float32), log_prob
        return out.to(torch.float32), action.to(torch.float32)

    def forward(self, obs, act=None):
        dist = self._distribution(obs)
        if act is not None:
            log_prob = dist.log_prob(act).sum(axis=-1)
            return dist, log_prob
        return dist
