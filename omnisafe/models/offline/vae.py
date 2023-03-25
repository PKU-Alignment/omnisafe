# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of VAE."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.normal import Normal

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VAE(nn.Module):
    """Class for VAE."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        latent_dim: Optional[int] = None,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ):
        """Initialize VAE.

        Args:
            obs_space (OmnisafeSpace): Observation space.
            act_space (OmnisafeSpace): Action space.
            hidden_sizes (list): List of hidden layer sizes.
            latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
        """

        nn.Module.__init__(self)

        if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
            self._obs_dim = obs_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim = act_space.shape[0]
        else:
            raise NotImplementedError

        if not latent_dim:
            latent_dim = self._act_dim * 2

        self._latent_dim = latent_dim

        self._encoder = build_mlp_network(
            sizes=[self._obs_dim + self._act_dim] + hidden_sizes + [self._latent_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._decoder = build_mlp_network(
            sizes=[self._obs_dim + self._latent_dim] + hidden_sizes + [self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def encode(self, obs: torch.Tensor, act: torch.Tensor) -> Normal:
        """Encode observation to latent space."""
        latent = self._encoder(torch.cat([obs, act], dim=-1))
        mean, log_std = torch.chunk(latent, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return Normal(mean, log_std.exp())

    def decode(self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent space to action."""
        if latent is None:
            latent = Normal(0, 1).sample([obs.shape[0], self.latent_dim]).to(obs.device)

        return self._decoder(torch.cat([obs, latent], dim=-1))

    def loss(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for VAE."""
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        recon_loss = nn.functional.mse_loss(pred_act, act)
        kl_loss = torch.distributions.kl.kl_divergence(dist, Normal(0, 1)).mean()
        return recon_loss, kl_loss

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, Normal, torch.Tensor]:
        """Forward function for VAE."""
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        return pred_act, dist, latent
