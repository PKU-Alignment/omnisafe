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
"""Implementation of ConstraintActorQCritic."""

from typing import NamedTuple, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.models.actor_q_critic import ActorQCritic
from omnisafe.models.critic.q_critic import QCritic


class ConstraintActorQCritic(ActorQCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

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
        *   -   Reward Q-Critic
            -   The value network, input is observation-action pair,
                output is reward value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the reward value of the observation.
        *   -   Cost Q-Critic
            -   The value network, input is observation-action pair,
                output is cost value.
                Choose the critic from the following options:
                :class:`QCritic`, :class:`VCritic`.
            -   Estimate the cost value of the observation.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        model_cfgs: NamedTuple,
    ) -> None:
        """Initialize ConstraintActorQCritic.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            standardized_obs: Whether to standardize the observation.
            model_cfgs: The model configurations.
        """

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_cfgs=model_cfgs,
        )
        self.cost_critic = QCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.ac_kwargs.val.hidden_sizes,
            activation=self.ac_kwargs.val.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=model_cfgs.shared_weights,
        )

    def step(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """Step function of the actor-critic model

        Input observation-action pair, output reward and cost value (from :class:`QCritic`) action,
        and its log probability (from :class`Actor`).

        .. note::
            The observation is standardized by the running mean and standard deviation.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to use deterministic action.
        """
        with torch.no_grad():
            raw_action, action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )
            value = self.critic(obs, action)[0]
            cost_value = self.cost_critic(obs, action)[0]

        return raw_action, action, value, cost_value, logp_a
