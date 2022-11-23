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
from torch.distributions.normal import Normal

from omnisafe.algos.models.actor import Actor
from omnisafe.algos.models.model_utils import build_mlp_network


class MLPGaussianActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        weight_initialization_mode,
        shared=None,
    ):
        super().__init__(obs_dim, act_dim, weight_initialization_mode)
        log_std = np.log(0.5) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=False)

        if shared is not None:  # use shared layers
            action_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.net = nn.Sequential(shared, action_head, nn.Identity())
        else:
            layers = [self.obs_dim] + list(hidden_sizes) + [self.act_dim]
            self.net = build_mlp_network(
                layers,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

    def dist(self, obs):
        mu = self.net(obs)
        return Normal(mu, self.std)

    def detach_dist(self, obs):
        mu = self.net(obs).detach()
        return Normal(mu, self.std.detach())

    def log_prob_from_dist(self, dist, act) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return dist.log_prob(act).sum(axis=-1)

    def predict(self, obs, deterministic=False):
        if deterministic == False:
            dist = self.dist(obs)
            action = dist.sample()
            logp_a = self.log_prob_from_dist(dist, action)
        else:
            action = self.net(obs)
            logp_a = torch.ones_like(action)  # avoid type conflicts at evaluation
        return action, logp_a

    def set_log_std(self, frac):
        """To support annealing exploration noise.
        frac is annealing from 1. to 0 over course of training"""
        assert 0 <= frac <= 1
        new_stddev = 0.499 * frac + 0.01  # annealing from 0.5 to 0.01
        log_std = np.log(new_stddev) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=False)

    @property
    def std(self):
        """Standard deviation of distribution."""
        return torch.exp(self.log_std)


if __name__ == '__main__':
    obs_dim = 10
    act_dim = 25
    hidden_sizes = [64, 64]
    activation = 'relu'
    weight_initialization_mode = 'kaiming_uniform'
    shared = None
    gaussianActor = MLPGaussianActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=None,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    out, logpro = gaussianActor(obs)
    assert isinstance(out, torch.distributions.normal.Normal) and logpro is None, 'Failed!'
    out, logpro = gaussianActor(obs, torch.tensor(act_dim, dtype=torch.float32))
    assert isinstance(out, torch.distributions.normal.Normal) and isinstance(
        logpro, torch.Tensor
    ), 'Failed!'

    dist = gaussianActor.dist(obs)
    assert isinstance(dist, torch.distributions.normal.Normal), 'Failed'
