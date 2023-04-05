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
"""Implementation of the Twin Delayed DDPG algorithm."""

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class TD3(DDPG):
    """The Twin Delayed DDPG (TD3) algorithm.

    References:
        - Title: Addressing Function Approximation Error in Actor-Critic Methods
        - Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        - URL: `TD3 <https://arxiv.org/abs/1802.09477>`_
    """

    def _init_model(self) -> None:
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update reward critic using TD3 algorithm.

        Args:
            obs (torch.Tensor): current observation
            act (torch.Tensor): current action
            reward (torch.Tensor): current reward
            done (torch.Tensor): current done signal
            next_obs (torch.Tensor): next observation

        Returns:
            None
        """
        with torch.no_grad():
            # set the update noise and noise clip.
            next_action = self._actor_critic.target_actor.predict(next_obs, deterministic=True)
            policy_noise = self._cfgs.algo_cfgs.policy_noise
            policy_noise_clip = self._cfgs.algo_cfgs.policy_noise_clip
            noise = (torch.randn_like(next_action) * policy_noise).clamp(
                -policy_noise_clip,
                policy_noise_clip,
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r)
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, target_q_value_r) + nn.functional.mse_loss(
            q2_value_r,
            target_q_value_r,
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )
