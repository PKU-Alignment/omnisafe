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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class GaussianActor(Actor):
    """Implementation of GaussianStdNetActor."""

    # pylint: disable-next=too-many-arguments, too-many-locals
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shared: nn.Module = None,
        scale_action: bool = False,
        clip_action: bool = False,
        std_learning: bool = True,
        std_init: float = 1.0,
        std_end: float = 1.0,
        std_annealing: bool = False,
    ) -> None:
        """Initialize GaussianStdNetActor.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            act_max (torch.Tensor): Maximum value of the action.
            act_min (torch.Tensor): Minimum value of the action.
            hidden_sizes (list): List of hidden layer sizes.
            activation (Activation): Activation function.
            output_activation (Activation): Activation function for the output layer.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared module.
            scale_action (bool): Whether to scale the action.
            clip_action (bool): Whether to clip the action.
            std_learning (bool): Whether to learn the standard deviation.
            std_init (float): Initial value of the standard deviation.
            std_end (float): Final value of the standard deviation.
            std_annealing (bool): Whether to anneal the standard deviation.
        """
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared
        )
        self.act_min = act_min
        self.act_max = act_max
        self.scale_action = scale_action
        self.clip_action = clip_action
        self.std_init = std_init
        self._std = std_init
        self.std_end = std_end
        self.std_annealing = std_annealing
        assert (
            self.act_min.size() == self.act_max.size()
        ), f'The size of act_min {self.act_min} and act_max {self.act_max} should be the same.'
        if std_annealing:
            assert (
                std_init > std_end
            ), 'If std_annealing is True, std_init should be greater than std_end.'
            assert not std_learning, 'If std_annealing is True, std_learning should be False.'
        if std_learning:
            assert not std_annealing, 'If std_learning is True, std_annealing should be False.'

        if shared is not None:
            mean_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, mean_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim],
                activation=activation,
                output_activation=output_activation,
                weight_initialization_mode=weight_initialization_mode,
            )
        self.logstd_layer = nn.Parameter(torch.zeros(1, act_dim), requires_grad=std_learning)

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
        mean = self.net(obs)
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        log_std = self.logstd_layer.expand_as(mean)
        std = torch.exp(log_std) * self._std

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
            out = mean.to(torch.float64)
        else:
            out = dist.rsample().to(torch.float64)

        if self.scale_action:
            # If the action scale is inf, stop scaling the action
            assert (
                not torch.isinf(self.act_min).any() and not torch.isinf(self.act_max).any()
            ), 'The action scale is inf, stop scaling the action.'
            self.act_min = self.act_min.to(mean.device)
            self.act_max = self.act_max.to(mean.device)
            action = self.act_min + (out + 1) / 2 * (self.act_max - self.act_min)
        else:
            action = out

        if self.clip_action:
            action = torch.clamp(action, self.act_min, self.act_max)

        if need_log_prob:
            log_prob = dist.log_prob(out).sum(axis=-1)
            return out.to(torch.float32), action.to(torch.float32), log_prob.to(torch.float32)
        return out.to(torch.float32), action.to(torch.float32)

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

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Get distribution of the action.
        Args:
            obs (torch.Tensor): Observation.
        """
        return self._distribution(obs)

    def set_std(self, proportion: float) -> float:
        """To support annealing exploration noise.

        Proportion is annealing from 1. to 0 over course of training.

        Args:
            proportion (float): proportion of annealing.
        """
        self._std = self.std_init * proportion + self.std_end * (1 - proportion)
