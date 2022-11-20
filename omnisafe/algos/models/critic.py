import torch
import torch.nn as nn

from omnisafe.algos.models.model_utils import build_mlp_network


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list, activation: str, shared=None) -> None:
        super().__init__()
        if shared is None:
            self.net = build_mlp_network([obs_dim] + hidden_sizes + [1], activation=activation)
        else:  # use shared layers
            value_head = nn.Linear(hidden_sizes[-1], 1)
            self.net = nn.Sequential(shared, value_head, nn.Identity())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Function Name.

        Args:
            obs: description

        Returns:
            description
        """
        return torch.squeeze(self.net(obs), -1)
