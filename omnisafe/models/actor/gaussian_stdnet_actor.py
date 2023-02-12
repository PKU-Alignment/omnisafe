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
"""Implementation of GaussianStdNetActor."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.typing import Activation
from omnisafe.utils.model import build_mlp_network


class GaussianStdNetActor(Actor):
    """Implementation of GaussianStdNetActor."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: Activation = 'kaiming_uniform',
        shared=None,
        scale_action=False,
        clip_action: bool = False,
    ):
        """Initialize GaussianStdNetActor."""
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max
        self.scale_action = scale_action
        self.clip_action = clip_action

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
            net = build_mlp_network(
                [obs_dim] + list(hidden_sizes),
                activation=activation,
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            std_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.mean = nn.Sequential(net, mean_head)
            self.log_std = nn.Sequential(net, std_head)
        self.net = nn.ModuleList([self.mean, self.log_std])

    def _distribution(self, obs):
        """Get distribution of the action.

        .. note::
            The term ``log_std`` is used to control the noise level of the policy,
            which is a trainable parameter.
            To avoid the policy to be too explorative,
            we use ``torch.clamp`` to limit the range of ``log_std``.

        Args:
            obs (torch.Tensor): Observation.
        """
        mean = self.mean(obs)
        log_std = self.log_std(obs)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict(self, obs, deterministic=False, need_log_prob=False):
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

        if self.scale_action:
            # If the action scale is inf, stop scaling the action
            assert (
                not torch.isinf(self.act_min).any() and not torch.isinf(self.act_max).any()
            ), 'The action scale is inf, stop scaling the action.'
            self.act_min = self.act_min.to(out.device)
            self.act_max = self.act_max.to(out.device)
            action = self.act_min + (out + 1) / 2 * (self.act_max - self.act_min)
        else:
            action = out

        if self.clip_action:
            action = torch.clamp(action, self.act_min, self.act_max)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            log_prob -= torch.log(1.00001 - torch.tanh(out) ** 2).sum(axis=-1)
            return out.to(torch.float32), action.to(torch.float32), log_prob.to(torch.float32)
        return out.to(torch.float32), action.to(torch.float32)

    def forward(self, obs, act=None):
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
