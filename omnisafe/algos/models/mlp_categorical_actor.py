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

from torch.distributions.categorical import Categorical

from omnisafe.algos.models.actor import Actor
from omnisafe.algos.models.model_utils import build_mlp_network


class MLPCategoricalActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        weight_initialization_mode,
        shared=None,
    ):
        super().__init__(obs_dim, act_dim, weight_initialization_mode, shared=shared)
        if shared is not None:
            raise NotImplementedError
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def dist(self, obs):

        logits = self.net(obs)
        return Categorical(logits=logits)

    def log_prob_from_dist(self, pi, act):

        return pi.log_prob(act)

    def predict(self, obs, deterministic=False):
        dist = self.dist(obs)
        if deterministic:
            a = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            a = dist.sample()
        logp_a = self.log_prob_from_dist(dist, a)
        return a, logp_a
