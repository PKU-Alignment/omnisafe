import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.algos.models.actor import Actor
from omnisafe.algos.models.model_utils import build_mlp_network


class OMLPGaussianActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        weight_initialization_mode,
        shared=None,
        act_limit=1,
    ):
        self.act_limit = act_limit
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
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def detach_dist(self, obs):
        mu = self.net(obs).detach()
        return Normal(mu, self.std.detach())

    def log_prob_from_dist(self, dist, act) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return dist.log_prob(act).sum(axis=-1)

    def predict(self, obs, determinstic=False):
        if determinstic == False:
            dist = self.dist(obs)
            action = dist.sample()
            logp_a = self.log_prob_from_dist(dist, action)
        else:
            action = self.net(obs)
            dist = self.dist(obs)
            logp_a = self.log_prob_from_dist(dist, action)
        action = torch.tanh(action)
        action = self.act_limit * action
        return action, logp_a

    def set_log_std(self, frac):
        """To support annealing exploration noise.
        frac is annealing from 1. to 0 over course of training"""
        assert 0 <= frac <= 1
        new_stddev = 0.499 * frac + 0.01  # annealing from 0.5 to 0.01
        log_std = np.log(new_stddev) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)

    @property
    def std(self):
        """Standard deviation of distribution."""
        return torch.exp(self.log_std)
