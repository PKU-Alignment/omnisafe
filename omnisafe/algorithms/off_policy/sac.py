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
"""Implementation of the Policy Gradient algorithm."""


import torch
from torch import optim
from torch.nn import functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SAC(DDPG):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init(self) -> None:
        super()._init()
        if self._cfgs.auto_alpha:
            self._target_entropy = -torch.prod(
                torch.Tensor(self._env.action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha = self._log_alpha.exp().item()
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha], lr=self._cfgs.model_cfgs.critic.lr
            )
            pass
        else:
            self._alpha = self._cfgs.alpha

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Value/reward_critic2')
        self._logger.register_key('Value/alpha')

    def _update_rewrad_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._target_actor_critic.actor.predict(next_obs, deterministic=False)
            next_action_log_prob = self._target_actor_critic.actor.log_prob(next_action)
            next_q_value = (
                torch.min(
                    self._target_actor_critic.reward_critic(next_obs, next_action)[0],
                    self._target_actor_critic.reward_critic(next_obs, next_action)[1],
                )
                - self._alpha * next_action_log_prob
            )
            target_q_value = reward + self._cfgs.gamma * (1 - done) * next_q_value
        q_values = self._actor_critic.reward_critic(obs, act)
        loss_critic1 = F.mse_loss(q_values[0], target_q_value)
        loss_critic2 = F.mse_loss(q_values[1], target_q_value)
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
                'Value/reward_critic1': q_values[0].mean().item(),
                'Value/reward_critic2': q_values[1].mean().item(),
            }
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        super()._update_actor(obs)
        if self._cfgs.auto_alpha:
            with torch.no_grad():
                log_prob = self._actor_critic.actor.log_prob(self._actor_critic.actor.predict(obs))
            alpha_loss = (-self._log_alpha * (log_prob + self._target_entropy)).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._alpha = self._log_alpha.exp().item()
            self._logger.store(
                **{
                    'Value/alpha': self._alpha,
                }
            )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        loss_q1 = self._actor_critic.reward_critic(obs, action)[0].mean()
        loss_q2 = self._actor_critic.reward_critic(obs, action)[1].mean()
        loss = self._alpha - torch.min(loss_q1, loss_q2)
        return loss

    def _log_zero(self) -> None:
        super()._log_zero()
        self._logger.store(
            **{
                'Value/reward_critic2': 0.0,
                'Value/alpha': self._alpha,
            }
        )
