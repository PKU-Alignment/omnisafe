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
"""Implementation of ConstraintActorQCritic."""

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorQCritic(ActorQCritic):
    """ConstraintActorQCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In ``omnisafe``, we combine the actor and critic into one this class.

    .. list-table::

        *   -   Model
            -   Description
            -   Function
        *   -   Actor
            -   The policy network, input is observation, output is action.
                Choose the actor from the following options:
                :class:`MLPActor`, :class:`CategoricalActor`, :class:`GaussianAnnealingActor`,
                :class:`GaussianLearningActor`, :class:`GaussianStdNetActor`, :class:`MLPCholeskyActor`.
            -   Choose the action based on the observation.
        *   -   Reward Critic
            -   The value network, input is observation,
                output is reward value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the reward value of the observation.
        *   -   Cost Critic
            -   The value network, input is observation,
                output is cost value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the cost value of the observation.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize ConstraintActorCritic."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.cost_critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('q')
        self.add_module('cost_critic', self.cost_critic)

        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=model_cfgs.critic.lr
        )

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool): Whether to use deterministic policy.

        Returns:
            action (torch.Tensor): Action.
            value_r (torch.Tensor): Reward value.
            value_c (torch.Tensor): Cost value.
            log_prob (torch.Tensor): Log probability of action.
        """
        with torch.no_grad():
            act = self.actor.predict(obs, deterministic=deterministic)

        return act

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool): Whether to use deterministic policy.

        Returns:
            action (torch.Tensor): Action.
            value_r (torch.Tensor): Reward value.
            value_c (torch.Tensor): Cost value.
            log_prob (torch.Tensor): Log probability of action.
        """
        return self.step(obs, deterministic=deterministic)