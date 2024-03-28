# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of ActorQCritic."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.base import Actor, Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ActorQCritic(nn.Module):
    """Class for ActorQCritic.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair. Output is reward value. |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
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
        """Initialize an instance of :class:`ActorQCritic`."""
        super().__init__()

        self.actor: Actor = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        ).build_actor(
            actor_type=model_cfgs.actor_type,
        )
        self.reward_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=model_cfgs.critic.num_critics,
            use_obs_encoder=False,
        ).build_critic(critic_type='q')
        self.target_reward_critic: Critic = deepcopy(self.reward_critic)
        for param in self.target_reward_critic.parameters():
            param.requires_grad = False
        self.target_actor: Actor = deepcopy(
            self.actor,
        )
        for param in self.target_actor.parameters():
            param.requires_grad = False
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)

        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: optim.Optimizer
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: optim.Optimizer
            self.reward_critic_optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

        self.actor_scheduler: LinearLR | ConstantLR
        if model_cfgs.linear_lr_decay:
            self.actor_scheduler = LinearLR(
                self.actor_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=epochs,
            )
        else:
            self.actor_scheduler = ConstantLR(
                self.actor_optimizer,
                factor=1.0,
                total_iters=epochs,
            )

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.actor.predict(obs, deterministic=deterministic)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        return self.step(obs, deterministic=deterministic)

    def polyak_update(self, tau: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        for param, target_param in zip(
            self.reward_critic.parameters(),
            self.target_reward_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
