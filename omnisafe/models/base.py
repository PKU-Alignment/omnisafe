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

import abc
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.typing import Activation, InitFunction


class Actor(abc.ABC, nn.Module):
    """A abstract class for actor.

    An actor approximates the policy function that maps observations to actions.
    Actor is parameterized by a neural network that takes observations as input,
    and outputs the mean and standard deviation of the action distribution.

    .. note::
        You can use this class to implement your own actor by inheriting it.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize the base actor.

        Args:
            obs_dim (int): observation dimension.
            act_dim (int): action dimension.
            hidden_sizes (list): hidden layer sizes.
            activation (Activation): activation function.
            weight_initialization_mode (InitFunction, optional): weight initialization mode.
                                                                Defaults to ``xavier_uniform``.
            shared (nn.Module, optional): shared module. Defaults to None.
        """
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes

    @abc.abstractmethod
    def _distribution(self, obs) -> Normal:
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
        """

    @abc.abstractmethod
    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Predict deterministic or stochastic action based on observation.

        - ``deterministic`` = ``True`` or ``False``

        When training the actor,
        one important trick to avoid local minimum is to use stochastic actions,
        which can simply be achieved by sampling actions from the distribution
        (set ``deterministic`` = ``False``).

        When testing the actor,
        we want to know the actual action that the agent will take,
        so we should use deterministic actions (set ``deterministic`` = ``True``).

        - ``need_log_prob`` = ``True`` or ``False``

        In some cases, we need to calculate the log probability of the action,
        which is used to calculate the loss of the actor.
        For example, in the case of Policy Gradient,
        the loss is defined as

        .. math::
            L = -\mathbb{E}_{s \sim p(s)} [\log p(a | s) A^R (s, a)]

        where :math:`p(s)` is the distribution of observation,
        :math:`p(a | s)` is the distribution of action,
        and :math:`\log p(a | s)` is the log probability of action under the distribution.,
        :math:`A^R (s, a)` is the advantage function.

        Args:
            obs (torch.Tensor): observation.
            deterministic (bool, optional): whether to predict deterministic action. Defaults to False.
            need_log_prob (bool, optional): whether to return log probability of action. Defaults to False.
        """


class Critic(abc.ABC, nn.Module):
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
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize the base critic.

        Args:
            obs_dim (int): observation dimension.
            act_dim (int): action dimension.
            hidden_sizes (list): hidden layer sizes.
            activation (Activation, optional): activation function. Defaults to 'relu'.
            weight_initialization_mode (InitFunction, optional): weight initialization mode.
                                                                Defaults to 'xavier_uniform'.
            shared (nn.Module, optional): shared module. Defaults to None.
        """
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor = None,
    ) -> Union[torch.Tensor, List]:
        """Forward function for critic.

        .. note::
            This forward function has two modes:
            - If ``act`` is not None, it will return the value of the observation-action pair.
            - If ``act`` is None, it will return the value of the observation.

        Args:
            obs (torch.Tensor): observation.
            act (torch.Tensor, optional): action. Defaults to None.
        """
