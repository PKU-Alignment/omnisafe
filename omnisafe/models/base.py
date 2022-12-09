# Copyright 2022 OmniSafe Team. All Rights Reserved.
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

import torch.nn as nn

from omnisafe.utils.model_utils import Activation, InitFunction


class Actor(abc.ABC, nn.Module):
    """A abstract class for actor."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation,
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes

    @abc.abstractmethod
    def _distribution(self, obs):
        """distribution of action.

        Args:
            obs (torch.Tensor): observation.

        Returns:
            torch.distributions.Distribution
        """

    @abc.abstractmethod
    def predict(self, obs, deterministic=False, need_log_prob=False):
        """predict deterministic or stochastic action based on observation.

        Args:
            obs (torch.Tensor): observation.
            determinstic (bool, optional): whether to predict deterministic action. Defaults to False.
            need_log_prob (bool, optional): whether to return log probability of action. Defaults to False.

        Returns:
            torch.Tensor: predicted action.
            torch.Tensor: log probability of action under the distribution.
        """

    def forward(self, obs, act=None):
        """forward function for actor.

        pi = p(a | s)
        usage 1: pi, logp_a = actor(obs, act)
        usage 2: pi = actor(obs)

        Args:
            obs (torch.Tensor): observation.
            act (torch.Tensor, optional): action. Defaults to None.

        Returns:
            torch.distributions.Distribution: distribution of action.
            torch.Tensor: log probability of action under the distribution.
        """
        dist = self._distribution(obs)
        if act is not None:
            logp_a = dist.log_prob(act)
            return dist, logp_a
        return dist


class Critic(abc.ABC, nn.Module):
    """A abstract class for critic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes

    @abc.abstractmethod
    def forward(self, obs):
        """forward function for critic.

        Args:
            obs (torch.Tensor): observation.

        Returns:
            torch.Tensor: value of observation.
        """
