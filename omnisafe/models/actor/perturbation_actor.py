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
"""Implementation of Perturbation Actor."""

from typing import List

import torch
from torch.distributions import Distribution

from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class PerturbationActor(Actor):
    """Class for Perturbation Actor.

    Perturbation Actor is used in offline algorithms such as ``BCQ`` and so on.
    Perturbation Actor is a combination of VAE and a perturbation network,
    algorithm BCQ uses the perturbation network to perturb the action predicted by VAE,
    which trained like behavior cloning.

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
        """Initialize an instance of :class:`PerturbationActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.vae = VAE(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.perturbation = build_mlp_network(
            sizes=[self._obs_dim + self._act_dim, *hidden_sizes, self._act_dim],
            activation=activation,
            output_activation='tanh',
            weight_initialization_mode=weight_initialization_mode,
        )
        self._phi = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def phi(self) -> float:
        """Return phi, which is the maximum perturbation."""
        return self._phi.item()

    @phi.setter
    def phi(self, phi: float) -> None:
        """Set phi. which is the maximum perturbation."""
        self._phi = torch.nn.Parameter(torch.tensor(phi, device=self._phi.device))

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action from observation.

        deterministic is not used in this method, it is just for compatibility.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to return deterministic action. Defaults to False.

        Returns:
            torch.Tensor: Action.
        """
        act = self.vae.predict(obs, deterministic)
        perturbation = self.perturbation(torch.cat([obs, act], dim=-1))
        return act + self._phi * perturbation

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError
