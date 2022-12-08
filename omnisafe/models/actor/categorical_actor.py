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
"""Implementation of categorical actor."""

import torch.nn as nn
from torch.distributions.categorical import Categorical

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class CategoricalActor(Actor):
    """Categorical actor."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation,
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared=None,
    ):
        """Categorical actor."""
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared=shared
        )
        if shared is not None:
            action_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, action_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def predict(self, obs, deterministic=False, need_log_prob=False):
        dist = self._distribution(obs)
        if deterministic:
            a = dist.probs.argmax(dim=-1)
        else:
            a = dist.sample().squeeze(dim=-1)
        if need_log_prob:
            logp_a = dist.log_prob(a)
            return a, logp_a
        return a
