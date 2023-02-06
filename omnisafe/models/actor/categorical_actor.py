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
"""Implementation of categorical actor."""

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from omnisafe.models.base import Actor
from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network


class CategoricalActor(Actor):
    """Implementation of CategoricalActor.

    A Categorical policy that uses a MLP to map observations to actions distributions.
    :class:`CategoricalActor` uses a single headed MLP,
    to predict the logits of the Categorical distribution.
    This class is an inherit class of :class:`Actor`.
    You can design your own Categorical policy by inheriting this class or :class:`Actor`.
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
        """Initialize CategoricalActor.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        super().__init__(
            obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode, shared=shared
        )
        if shared is not None:
            action_head = build_mlp_network(
                sizes=[hidden_sizes[-1], act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net = nn.Sequential(shared, action_head)
        else:
            self.net = build_mlp_network(
                [obs_dim] + list(hidden_sizes) + [act_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """Get distribution of the action.

        .. note::
            This function is used to get the distribution of the action.
            It is used to sample actions and compute log probabilities.

        Args:
            obs (torch.Tensor): Observation.
        """
        logits = self.net(obs)
        return Categorical(logits=logits)

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
        For example, in the case of continuous action space,
        the loss can be calculated as:

        .. math::
            L = -\mathbb{E}_{s \sim p(s)} [\log p(a | s) A^R (s, a)]

        where :math:`p(s)` is the distribution of observation,
        :math:`p(a | s)` is the distribution of action,
        and :math:`\log p(a | s)` is the log probability of action under the distribution.

        Args:
            obs (torch.Tensor): observation.
            deterministic (bool, optional): whether to predict deterministic action. Defaults to False.
            need_log_prob (bool, optional): whether to return log probability of action. Defaults to False.
        """
        dist = self._distribution(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        action = action.unsqueeze(0)
        if need_log_prob:
            logp_a = dist.log_prob(action)
            return action, action, logp_a
        return action, action
