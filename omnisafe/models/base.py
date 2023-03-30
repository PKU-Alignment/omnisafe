# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""This module contains some base abstract classes for the models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Distribution

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace


class Actor(ABC, nn.Module):
    """A abstract class for actor.

    An actor approximates the policy function that maps observations to actions.
    Actor is parameterized by a neural network that takes observations as input,
    and outputs the mean and standard deviation of the action distribution.

    .. note::
        You can use this class to implement your own actor by inheriting it.

    Attributes:
        _obs_space (OmnisafeSpace): observation space.
        _act_space (OmnisafeSpace): action space.
        _weight_initialization_mode (InitFunction): weight initialization mode.
        _activation (Activation): activation function.
        _hidden_sizes (list): hidden layer sizes.
        _current_dist (Distribution): current distribution.
        _after_inference (bool): whether the actor has been used to sample actions.
        _obs_dim (int): observation dimension.
        _act_dim (int): action dimension.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize the base actor.

        Args:
            obs_space (OmnisafeSpace): observation space.
            act_space (OmnisafeSpace): action space.
            hidden_sizes (list): hidden layer sizes.
            activation (Activation): activation function.
            weight_initialization_mode (InitFunction, optional): weight initialization mode.
                                                                Defaults to ``kaiming_uniform``.
            shared (nn.Module, optional): shared module. Defaults to None.
        """
        nn.Module.__init__(self)
        self._obs_space = obs_space
        self._act_space = act_space
        self._weight_initialization_mode = weight_initialization_mode
        self._activation = activation
        self._hidden_sizes = hidden_sizes

        self._current_dist: Distribution
        self._after_inference: bool = False

        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim = self._obs_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(self._act_space, spaces.Box) and len(self._act_space.shape) == 1:
            self._act_dim = self._act_space.shape[0]
        else:
            raise NotImplementedError

    @abstractmethod
    def _distribution(self, obs: torch.Tensor) -> Distribution:
        r"""Return the distribution of action.

        An actor generates a distribution, which is used to sample actions during training.
        When training, the mean and the variance of the distribution are used to calculate the loss.
        When testing, the mean of the distribution is used directly as actions.

        For example, if the action is continuous, the actor can generate a Gaussian distribution.

        .. math::
            p(a | s) = N(a | \mu(s), \sigma(s))

        where :math:`\mu(s)` and :math:`\sigma(s)` are the mean and standard deviation of the distribution.

        .. warning::
            The distribution is a private method, which is only used to sample actions during training.
            You should not use it directly in your code,
            instead, you should use the public method ``predict`` to sample actions.

        Args:
            obs (torch.Tensor): observation.

        Returns:
            Distribution: the distribution of action.
        """

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Distribution:
        r"""Return the distribution of action.

        Args:
            obs (torch.Tensor): observation.
        """

    @abstractmethod
    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        r"""Predict deterministic or stochastic action based on observation.

        - ``deterministic`` = ``True`` or ``False``

        When training the actor,
        one important trick to avoid local minimum is to use stochastic actions,
        which can simply be achieved by sampling actions from the distribution
        (set ``deterministic`` = ``False``).

        When testing the actor,
        we want to know the actual action that the agent will take,
        so we should use deterministic actions (set ``deterministic`` = ``True``).

        .. math::
            L = -\mathbb{E}_{s \sim p(s)} [\log p(a | s) A^R (s, a)]

        where :math:`p(s)` is the distribution of observation,
        :math:`p(a | s)` is the distribution of action,
        and :math:`\log p(a | s)` is the log probability of action under the distribution.,
        :math:`A^R (s, a)` is the advantage function.

        Args:
            obs (torch.Tensor): observation.
            deterministic (bool, optional): whether to predict deterministic action. Defaults to False.
        """

    @abstractmethod
    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        r"""Return the log probability of action under the distribution.

        ``log_prob`` only can be called after calling ``predict`` or ``forward``.

        Args:
            obs (torch.Tensor): observation.
            act (torch.Tensor): action.

        Returns:
            torch.Tensor: the log probability of action under the distribution.
        """


class Critic(ABC, nn.Module):
    """A abstract class for critic.

    A critic approximates the value function that maps observations to values.
    Critic is parameterized by a neural network that takes observations as input,
    (Q critic also takes actions as input) and outputs the value of the observation.

    .. note::
        Omnisafe provides two types of critic:
        Q critic (Input = ``observation`` + ``action`` , Output = ``value``),
        and V critic (Input = ``observation`` , Output = ``value``).
        You can also use this class to implement your own actor by inheriting it.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
    ) -> None:
        """Initialize the base critic.

        Args:
            obs_space (OmnisafeSpace): observation space.
            act_space (OmnisafeSpace): action space.
            hidden_sizes (list): hidden layer sizes.
            activation (Activation, optional): activation function. Defaults to 'relu'.
            weight_initialization_mode (InitFunction, optional): weight initialization mode.
                                                                Defaults to 'kaiming_uniform'.
            shared (nn.Module, optional): shared module. Defaults to None.
        """
        nn.Module.__init__(self)
        self._obs_space = obs_space
        self._act_space = act_space
        self._weight_initialization_mode = weight_initialization_mode
        self._activation = activation
        self._hidden_sizes = hidden_sizes
        self._num_critics = num_critics
        self._use_obs_encoder = use_obs_encoder

        if isinstance(self._obs_space, spaces.Box) and len(self._obs_space.shape) == 1:
            self._obs_dim = self._obs_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(self._act_space, spaces.Box) and len(self._act_space.shape) == 1:
            self._act_dim = self._act_space.shape[0]
        else:
            raise NotImplementedError
