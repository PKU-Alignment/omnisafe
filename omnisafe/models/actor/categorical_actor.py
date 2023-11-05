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
"""Implementation of CategoricalActor."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class CategoricalActor(Actor):
    """Implementation of CategoricalActor.

    CategoricalActor is an actor suitable for discrete action. It is used in
    discrete action space environment such as ``CartPole-v1`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _current_dist: Categorical

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`CategoricalActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.logits: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            Categorical distribution over actions based on the actor's logits.
        """
        logits = self.logits(obs)
        return Categorical(logits=logits)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action based on given observations.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the action with highest probability.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The action with highest probability if deterministic is True,
            otherwise a sampled action from the distribution.
        """
        self._current_dist = self._distribution(obs=obs)
        self._after_inference = True
        if deterministic:
            return torch.argmax(self._current_dist.logits, dim=0, keepdim=False)
        return self._current_dist.sample()

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
        return self._current_dist.log_prob(act)
