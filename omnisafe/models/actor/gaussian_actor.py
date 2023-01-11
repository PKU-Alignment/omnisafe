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
"""Implementation of GaussianStdNetActor."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class GaussianActor(Actor):
    """Implementation of GaussianStdNetActor.

    A Gaussian policy that uses a MLP to map observations to actions distributions.
    :class:`GaussianStdNetActor` uses a double headed MLP to predict the mean and log standard deviation
    of the Gaussian distribution.
    This class is an inherit class of :class:`Actor`.
    You can design your own Gaussian policy by inheriting this class or :class:`Actor`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shared: nn.Module = None,
        log_std=-0.5,
    ) -> None:
        """Initialize GaussianStdNetActor.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            act_max (torch.Tensor): Action maximum value.
            act_min (torch.Tensor): Action minimum value.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max

        if shared is not None:
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            std_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.mean = nn.Sequential(shared, mean_head)
            self.log_std = nn.Sequential(shared, std_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes),
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
            self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * log_std)
            nn.init.kaiming_uniform_(self.mean_layer.weight, a=np.sqrt(5))
            nn.init.constant_(self.mean_layer.bias, 0)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get distribution of the action.

        .. note::
            The term ``log_std`` is used to control the noise level of the policy,
            which is a trainable parameter.
            To avoid the policy to be too explorative,
            we use ``torch.clamp`` to limit the range of ``log_std``.

        Args:
            obs (torch.Tensor): Observation.
        """
        mean, std = self.get_mean_std(obs)
        return Normal(mean, std)

    def get_mean_std(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and std of the action.

        Args:
            obs (torch.Tensor): Observation.
        """
        out = self.net(obs)
        mean = self.mean_layer(out)
        # mean = self.act_max * torch.tanh(self.mean_layer(out))
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        log_std = self.logstd_layer.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of the action.

        Args:
            obs (torch.Tensor): Observation.
            action (torch.Tensor): Action.
        """
        dist = self._distribution(obs)
        return dist.log_prob(action).sum(axis=-1)

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Predict action given observation.

        .. note::
            The action is scaled to the action space by:

            .. math::
                a = a_{min} + \frac{a + 1}{2} \times (a_{max} - a_{min})

            where :math:`a` is the action predicted by the policy,
            :math:`a_{min}` and :math:`a_{max}` are the minimum and maximum values of the action space.
            After scaling, the action is clipped to the range of :math:`[a_{min}, a_{max}]`.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool): Whether to use deterministic policy.
        """
        mean, std = self.get_mean_std(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = torch.normal(mean, std)
        if need_log_prob:
            log_prob = dist.log_prob(action).sum(axis=-1)
            return action, log_prob
        return action

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function for actor.

        .. note::
            This forward function has two modes:
            - If ``act`` is not None, it will return the distribution and the log probability of action.
            - If ``act`` is None, it will return the distribution.

        Args:
            obs (torch.Tensor): observation.
            act (torch.Tensor, optional): action. Defaults to None.
        """
        dist = self._distribution(obs)
        if act is not None:
            log_prob = dist.log_prob(act).sum(axis=-1)
            return dist, log_prob
        return dist
