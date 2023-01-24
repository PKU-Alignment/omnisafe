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
"""Implementation of GaussianLearningActor."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class GaussianLearningActor(Actor):
    """Implementation of GaussianLearningActor.
    A Gaussian policy that uses a MLP to map observations to actions distributions.
    :class:`GaussianLearningActor` use a neural network,
    to output the mean of action distribution,
    while the ``_std`` is a learnable tensor served as the variance.
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
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared=None,
        satrt_std: float = 1.0,
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
        self.start_std = satrt_std
        self._std = nn.Parameter(
            self.start_std * torch.ones(self.act_dim, dtype=torch.float32), requires_grad=True
        )

        self.act_min = act_min
        self.act_max = act_max

        if shared is not None:
            action_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation='tanh',
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, action_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim],
                activation=activation,
                output_activation='tanh',
                weight_initialization_mode=weight_initialization_mode,
            )

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get distribution of the action.
        .. note::
            The mean of the distribution is the output of the network,
            while the variance is the learnable tensor ``_std``.
        Args:
            obs (torch.Tensor): Observation.
        """
        mean = self.net(obs)
        return Normal(mean, self._std)

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
        dist = self._distribution(obs)
        if deterministic:
            out = dist.mean
        else:
            out = dist.rsample()

        action = torch.clamp(out, -1, 1)
        action = self.act_min + (action + 1) * 0.5 * (self.act_max - self.act_min)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            return out, log_prob
        return out

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
            act = 2 * (act - self.act_min) / (self.act_max - self.act_min) - 1
            log_prob = dist.log_prob(act).sum(axis=-1)
            return dist, log_prob
        return dist

    @property
    def std(self) -> float:
        """Return the current std."""
        return self._std.mean().item()
