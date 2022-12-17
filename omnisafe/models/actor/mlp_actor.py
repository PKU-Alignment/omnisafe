import numpy as np
import torch
from torch import nn
from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import build_mlp_network
from omnisafe.utils.model_utils import Activation, InitFunction
from torch.distributions.normal import Normal


class MLPActor(Actor):
    """A abstract class for actor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_noise,
        act_limit,
        hidden_sizes: list,
        activation: Activation,
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation)
        self.act_limit = act_limit
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
        return self.act_limit * self.net(obs)

    def predict(self, obs, deterministic=False, need_log_prob=False):
        if deterministic:
            action = self.act_limit * self.net(obs)
        else:
            action = self.act_limit * self.net(obs)
            action += self.act_noise * np.random.randn(self.act_dim)
        return action.to(torch.float32), torch.tensor(1, dtype=torch.float32)
