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
"""Implementation of the Twin Delayed DDPG algorithm."""


import torch
from torch.nn import functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class TD3(DDPG):
    """The Twin Delayed DDPG (TD3) algorithm.

    References:
        - Title: Addressing Function Approximation Error in Actor-Critic Methods
        - Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        - URL: `TD3 <https://arxiv.org/abs/1802.09477>`_
    """

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Value/reward_critic_2')

    def _update_rewrad_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
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
            # Set the update noise and noise clip.
            self._actor_critic.actor.noise = self._cfgs.policy_noise
            self._actor_critic.actor.noise_clip = self._cfgs.policy_noise_clip
            next_action = self._target_actor_critic.actor.predict(next_obs, deterministic=False)
            next_q_value_r = torch.min(
                self._target_actor_critic.reward_critic(next_obs, next_action)[0],
                self._target_actor_critic.reward_critic(next_obs, next_action)[1],
            )
            target_q_value_r = reward + self._cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r_list = self._actor_critic.reward_critic(obs, act)
        loss_critic1 = F.mse_loss(q_value_r_list[0], target_q_value_r)
        loss_critic2 = F.mse_loss(q_value_r_list[1], target_q_value_r)
        loss = loss_critic1 + loss_critic2
        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic_1': q_value_r_list[0].mean().item(),
                'Value/reward_critic_2': q_value_r_list[1].mean().item(),
            }
        )

    def _log_zero(self) -> None:
        super()._log_zero()
        self._logger.store(
            **{
                'Value/reward_critic_2': 0.0,
            }
        )
