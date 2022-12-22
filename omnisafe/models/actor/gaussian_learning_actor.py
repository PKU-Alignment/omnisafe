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
"""Implementation of GaussianLearningActor."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import build_mlp_network


class GaussianLearningActor(Actor):
    """Implementation of GaussianLearningActor"""

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
        satrt_std: float = 0.5,
    ):
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.start_std = satrt_std
        self._std = nn.Parameter(self.start_std * torch.ones(self.act_dim, dtype=torch.float32))

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

        action = torch.clamp(out, -1, 1)
        action = self.act_min + (action + 1) * 0.5 * (self.act_max - self.act_min)

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

    @property
    def std(self):
        """Return the current std."""
        return self._std.mean().item()
