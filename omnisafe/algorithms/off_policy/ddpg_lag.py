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
"""Implementation of the Lagrangian version of Deep Deterministic Policy Gradient algorithm."""


import torch
from torch.nn import functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class DDPGLag(DDPG):
    """The Lagrangian version of Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        loss_r = -self._actor_critic.reward_critic(obs, action)[0].mean()
        loss_c = (
            self._lagrange.lagrangian_multiplier
            * self._actor_critic.cost_critic(obs, action)[0].mean()
        )
        return loss_r + loss_c

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        # cost=torch.ones_like(cost)*torch.mean(cost)
        with torch.no_grad():
            next_action = self._target_actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._target_actor_critic.cost_critic(next_obs, next_action)[0]
            target_q_value_c = cost + self._cfgs.gamma * (1 - done) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, act)[0]
        loss = F.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._lagrange.update_lagrange_multiplier(q_value_c.mean().item())

        self._logger.store(
            **{
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            }
        )

    def _log_zero(self) -> None:
        super()._log_zero()
        self._logger.store(
            **{
                'Loss/Loss_cost_critic': 0.0,
                'Value/cost_critic': 0.0,
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            }
        )
