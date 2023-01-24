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
"""Implementation of MLPActor."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions.normal import Normal

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class MLPActor(Actor):
    """Implementation of MLPActor.
    Different from gaussian actor,
    :class:`MLPActor` uses a multi-layer perceptron (MLP) to map observations directly to actions.
    This class is an inherit class of :class:`Actor`,
    you can design your own actor by inheriting this class or :class:`Actor`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_noise: float,
        act_max: float,
        act_min: float,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize the actor network.
        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            act_noise (float): Action noise.
            act_max (float): Action maximum value.
            act_min (float): Action minimum value.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        super().__init__(obs_dim, act_dim, hidden_sizes, activation)
        self.act_max = act_max
        self.act_min = act_min
        self.act_noise = act_noise
        self._std = 0.5 * torch.ones(self.act_dim, dtype=torch.float32)

        if shared is not None:  # use shared layers
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
        """Get the distribution of actor.
        Actually, this function is not used in this class.
        Args:
            obs (torch.Tensor): Observation.
        """
        mean = self.net(obs)
        return Normal(mean, self._std)

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of actor.
        .. note::
            :class:`_distribution` is a private function,
            which can not be called outside the class.
            You can use this function to get the distribution of actor.
        Args:
            obs (torch.Tensor): Observation.
        """
        return self._distribution(obs)

    def forward(
        self,
        obs: torch.Tensor,
        act: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.
        The forward action of actor network needs to be scaled to the action space limits.
        Args:
            obs (torch.Tensor): Observation.
            act (Optional[torch.Tensor], optional): Action. Defaults to None.
        """
        # Return output from network scaled to action space limits.
        return self.act_max * self.net(obs)

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Predict deterministic or stochastic action based on observation.
        - ``deterministic`` = ``True`` or ``False``.
        When training the actor,
        one important trick to avoid local minimum is to use stochastic actions,
        which can simply be achieved by sampling actions from the distribution
        (set ``deterministic`` = ``False``).
        When testing the actor,
        we want to know the actual action that the agent will take,
        so we should use deterministic actions (set ``deterministic`` = ``True``).
        - ``need_log_prob`` = ``True`` or ``False``.
        In some cases, we need to calculate the log probability of the action,
        which is used to calculate the loss of the actor.
        For example, in the case of continuous action space,
        the loss can be calculated as:
        .. math::
            L = -\mathbb{E}_{s \sim p(s)} [\log p(a | s) A^R (s, a)]
        where :math:`p(s)` is the distribution of observation,
        :math:`p(a | s)` is the distribution of action,
        and :math:`\log p(a | s)` is the log probability of action under the distribution.
        .. note::
            Specifically, the final action need to be clipped to the action space limits,
            by :meth:`torch.clamp` function.
        Args:
            obs (torch.Tensor): observation.
            deterministic (bool, optional): whether to predict deterministic action. Defaults to False.
            need_log_prob (bool, optional): whether to return log probability of action. Defaults to False.
        """
        if deterministic:
            action = self.act_max * self.net(obs)
        else:
            action = self.act_max * self.net(obs)
            action += self.act_noise * torch.rand(self.act_dim)

        action = torch.clamp(action, self.act_min, self.act_max)
        if need_log_prob:
            return (
                action.to(torch.float32),
                action.to(torch.float32),
                torch.tensor(1, dtype=torch.float32),
            )

        return action.to(torch.float32), action.to(torch.float32)
