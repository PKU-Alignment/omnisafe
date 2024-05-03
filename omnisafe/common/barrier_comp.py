# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Implementation of Compensator Used in Control Barrier Function."""


from __future__ import annotations

import torch
from torch import optim

from omnisafe.utils.config import Config
from omnisafe.utils.model import build_mlp_network


class BarrierCompensator(torch.nn.Module):
    """A module that represents a barrier compensator using a multi-layer perceptron (MLP) network.

    This module is designed to compute actions based on observations, with the intention of compensating for
    potential barriers in a control system or a similar application. It is built upon a configurable MLP network
    and trained using an optimization routine.

    Attributes:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        _cfgs (Config): Configuration parameters for the MLP network and training.
        model (torch.nn.Module): The MLP network.
        optimizer (torch.optim.Optimizer): The optimizer for training the network.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        cfgs (Config): Configuration parameters for the network and training.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfgs: Config) -> None:
        """Initialize the action compensator."""
        super().__init__()
        self._cfgs: Config = cfgs
        self.model: torch.nn.Module = build_mlp_network(
            sizes=[obs_dim, *self._cfgs.hidden_sizes, act_dim],
            activation=self._cfgs.activation,
            weight_initialization_mode=self._cfgs.weight_initialization_mode,
        )
        self.optimizer: optim.Adam = optim.Adam(self.parameters(), lr=self._cfgs.lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate the sum of previous compensating actions.

        Args:
            obs (torch.Tensor): The input observation.

        Returns:
            torch.Tensor: The estimation of previous compensating actions.
        """
        return self.model(obs)

    def update(
        self,
        observation: torch.Tensor,
        approx_compensating_act: torch.Tensor,
        compensating_act: torch.Tensor,
    ) -> torch.Tensor:
        """Train the barrier compensator model.

        This method updates the model parameters to minimize the difference between the model's output and the
        target, which is a combination of approximate compensating action and compensating action.

        Args:
            observation (torch.Tensor): The observation data.
            approx_compensating_act (torch.Tensor): The approximate compensating action.
            compensating_act (torch.Tensor): The actual compensating action.

        Returns:
            torch.Tensor: The loss after training.
        """
        # Train the model
        for _ in range(self._cfgs.update_iters):
            target = approx_compensating_act + compensating_act
            self.optimizer.zero_grad()
            loss = torch.pow((self(observation) - target), 2).mean()
            loss.backward()
            self.optimizer.step()

        return loss
