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
"""Implementation of Perturbation Actor used in BCQ."""

from typing import List

import torch
from torch.distributions import Distribution

from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class PerturbationActor(Actor):
    """Class for Perturbation Actor."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize Perturbation Actor.

        Args:
            obs_space (OmnisafeSpace): Observation space.
            act_space (OmnisafeSpace): Action space.
            hidden_sizes (list): List of hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
        """
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.vae = VAE(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.perturbation = build_mlp_network(
            sizes=[self._obs_dim + self._act_dim, *hidden_sizes] + [self._act_dim],
            activation=activation,
            output_activation='tanh',
            weight_initialization_mode=weight_initialization_mode,
        )
        self._phi = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def phi(self) -> float:
        """Return phi."""
        return self._phi.item()

    @phi.setter
    def phi(self, phi: float) -> None:
        """Set phi."""
        self._phi = torch.nn.Parameter(torch.tensor(phi, device=self._phi.device))

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action from observation."""
        act = self.vae.predict(obs, deterministic)
        perturbation = self.perturbation(torch.cat([obs, act], dim=-1))
        return act + self._phi * perturbation

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Predict action from observation."""
        raise NotImplementedError

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Predict action from observation."""
        raise NotImplementedError
