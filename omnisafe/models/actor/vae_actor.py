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
from torch.distributions import Distribution, Normal

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VAE(Actor):
    """Class for VAE.

    VAE is a variational auto-encoder. It is used in offline algorithms such as ``BCQ`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list): List of hidden layer sizes.
        latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`VAE`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self._latent_dim = self._act_dim * 2

        self._encoder = build_mlp_network(
            sizes=[self._obs_dim + self._act_dim, *hidden_sizes, self._latent_dim * 2],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._decoder = build_mlp_network(
            sizes=[self._obs_dim + self._latent_dim, *hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.add_module('encoder', self._encoder)
        self.add_module('decoder', self._decoder)

    def encode(self, obs: torch.Tensor, act: torch.Tensor) -> Normal:
        """Encode observation to latent distribution.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Normal: Latent distribution.
        """
        latent = self._encoder(torch.cat([obs, act], dim=-1))
        mean, log_std = torch.chunk(latent, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return Normal(mean, log_std.exp())

    def decode(self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent vector to action.

        When ``latent`` is None, sample latent vector from standard normal distribution.

        Args:
            obs (torch.Tensor): Observation.
            latent (Optional[torch.Tensor], optional): Latent vector. Defaults to None.

        Returns:
            torch.Tensor: Action.
        """
        if latent is None:
            latent = Normal(0, 1).sample([obs.shape[0], self._latent_dim]).to(obs.device)

        return self._decoder(torch.cat([obs, latent], dim=-1))

    def loss(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for VAE.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .
        """
        dist = self.encode(obs, act)
        latent = dist.rsample()
        pred_act = self.decode(obs, latent)
        recon_loss = nn.functional.mse_loss(pred_act, act)
        kl_loss = torch.distributions.kl.kl_divergence(dist, Normal(0, 1)).mean()
        return recon_loss, kl_loss

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def predict(  # pylint: disable=unused-argument
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Predict the action given observation.

        deterministic if not used in VAE model. VAE actor's default behavior is stochastic,
        sampling from the latent standard normal distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            torch.Tensor: Predicted action.
        """
        return self.decode(obs)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError
