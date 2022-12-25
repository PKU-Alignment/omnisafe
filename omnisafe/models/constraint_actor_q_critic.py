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

import torch

from omnisafe.models.actor_q_critic import ActorQCritic
from omnisafe.models.critic.q_critic import QCritic


class ConstraintActorQCritic(ActorQCritic):
    """ConstraintActorQCritic is a wrapper around ActorQCritic that adds a cost critic to the model."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        model_cfgs,
    ):
        """Initialize ConstraintActorQCritic."""

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            standardized_obs=standardized_obs,
            shared_weights=model_cfgs.shared_weights,
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

    def step(self, obs, deterministic=False):
        """
        If training, this includes exploration noise!
        Expects that obs is not pre-processed.
        Args:
            obs, description
        Returns:
            action, value, log_prob(action)
        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )
            value = self.critic(obs, action)[0]
            cost_value = self.cost_critic(obs, action)[0]

        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            cost_value.cpu().numpy(),
            logp_a.cpu().numpy(),
        )
