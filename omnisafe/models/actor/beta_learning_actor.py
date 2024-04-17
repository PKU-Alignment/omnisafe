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
"""Implementation of BetaLearningActor."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Distribution, Beta

from omnisafe.models.actor.gaussian_actor import GaussianActor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network
from omnisafe.models.base import Actor


# pylint: disable-next=too-many-instance-attributes
class BetaLearningActor(Actor):


    _current_dist: Beta
    
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`GaussianLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        
        self.mean: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, self._hidden_sizes[0], self._hidden_sizes[0]],
            activation=activation,
            output_activation='tanh',
            weight_initialization_mode=weight_initialization_mode,
        )
        
        self.alpha_net: nn.Module = build_mlp_network(
            sizes=[self._hidden_sizes[-1], self._act_dim],
            activation='identity',
            output_activation='softplus',
            weight_initialization_mode=weight_initialization_mode,
        )
        
        self.beta_net: nn.Module = build_mlp_network(
            sizes=[self._hidden_sizes[-1], self._act_dim],
            activation='identity',
            output_activation='softplus',
            weight_initialization_mode=weight_initialization_mode,
        )
        
    def _distribution(self, obs: torch.Tensor) -> Beta:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        mean = self.mean(obs)
        alphas = 1.0+self.alpha_net(mean)
        betas = 1.0+self.beta_net(mean)
        return Beta(alphas, betas)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.log_prob(act).sum(axis=-1)

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return 1.0

    @std.setter
    def std(self, std: float) -> None:
        pass
