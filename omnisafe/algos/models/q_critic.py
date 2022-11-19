import torch
import torch.nn as nn

from omnisafe.algos.models.model_utils import build_mlp_network


class Q_Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: str,
        shared=None,
    ) -> None:
        super().__init__()
        if shared is None:
            self.net1 = build_mlp_network([obs_dim] + [hidden_sizes[0]], activation, 'relu')
            self.net2 = build_mlp_network(
                [hidden_sizes[0] + act_dim] + [hidden_sizes[1]] + [1], activation=activation
            )
        else:  # use shared layers
            value_head = nn.Linear(hidden_sizes[-1], 1)
            self.net = nn.Sequential(shared, value_head, nn.Identity())

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """Function Name.

        Args:
            obs: description
            act

        Returns:
            description
        """
        v = self.net1(obs)
        data = torch.cat([v, act], dim=-1)
        v = self.net2(data)
        return torch.squeeze(v, -1)  # Critical to ensure q has right shape.
