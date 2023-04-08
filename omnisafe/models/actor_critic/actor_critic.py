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
"""Implementation of ActorCritic."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.schedule import PiecewiseSchedule, Schedule


class ActorCritic(nn.Module):
    """Class for ActorCritic.

    In ``omnisafe``, we combine the actor and critic into one this class.

    .. list-table::

        *   -   Model
            -   Description
            -   Function
        *   -   Actor
            -   The policy network, input is observation, output is action.
                Choose the actor from the following options:
                :class:`MLPActor`, :class:`GaussianSACActor`,
                :class:`GaussianLearningActor`.
            -   Choose the action based on the observation.
        *   -   Value Critic
            -   The value network, input is observation, output is reward value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the reward value of the observation.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize ActorCritic."""
        super().__init__()
        self.actor = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        ).build_actor(actor_type=model_cfgs.actor_type)
        self.reward_critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic(critic_type='v')
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)

        if model_cfgs.actor.lr is not None:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
        if model_cfgs.actor.lr is not None:
            self.actor_scheduler: LinearLR | ConstantLR
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                    verbose=True,
                )
            else:
                self.actor_scheduler = ConstantLR(
                    self.actor_optimizer,
                    factor=1.0,
                    total_iters=epochs,
                    verbose=True,
                )

            self.std_schedule: Schedule

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs: The observation.
            deterministic: Whether to use deterministic action. default: False.

        Returns:
            The action, value_r, and log_prob.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            act = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(act)
        return act, value_r[0], log_prob

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs: The observation.
            deterministic: Whether to use deterministic action. default: False.

        Returns:
            The action, value_r, and log_prob.
        """
        return self.step(obs, deterministic=deterministic)

    def set_annealing(self, epochs: list[int], std: list[float]) -> None:
        """Set the annealing mode for the actor.

        Args:
            annealing: Whether to use annealing mode.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.std_schedule = PiecewiseSchedule(
            endpoints=list(zip(epochs, std)),
            outside_value=std[-1],
        )

    def annealing(self, epoch: int) -> None:
        """Set the annealing mode for the actor.

        Args:
            epoch: The current epoch.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.actor.std = self.std_schedule.value(epoch)
