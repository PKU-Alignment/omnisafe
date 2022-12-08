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

import torch

from omnisafe.models.actor_critic import ActorCritic
from omnisafe.models.critic import CriticBuilder


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        scale_rewards: bool,
        # shared_weights: bool,
        # ac_kwargs: dict,
        # weight_initialization_mode='kaiming_uniform',
        model_cfgs,
    ) -> None:
        ActorCritic.__init__(
            self,
            observation_space,
            action_space,
            standardized_obs,
            scale_rewards,
            model_cfgs,
        )

        critic_builder = CriticBuilder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.ac_kwargs.val.hidden_sizes,
            activation=self.ac_kwargs.val.activation,
            weight_initialization_mode=self.model_cfgs.weight_initialization_mode,
            shared=self.shared,
        )
        self.cost_critic = critic_builder.build_critic('v')

    def step(self, obs, deterministic=False):
        """Produce action, value, log_prob(action).
        If training, this includes exploration noise!

        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: do the updates at the end of batch!
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            value = self.reward_critic(obs)
            cost_value = self.cost_critic(obs)

            action, logp_a = self.actor.predict(
                obs, deterministic=deterministic, need_log_prob=True
            )

        return action.numpy(), value.numpy(), cost_value.numpy(), logp_a.numpy()
