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
"""Implementation of BCQ-Lag."""

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.bcq import BCQ
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.critic.critic_builder import CriticBuilder


@registry.register
class BCQLag(BCQ):
    """Batch-Constrained Deep Reinforcement Learning with Lagrange Multiplier.

    References:
        - Title: Off-Policy Deep Reinforcement Learning without Exploration
        - Author: Fujimoto, ScottMeger, DavidPrecup, Doina.
        - URL: `https://arxiv.org/abs/1812.02900`
    """

    def _init_log(self) -> None:
        """Log the BCQLag specific information.

        +----------------------------+---------------------------------------------------------+
        | Things to log              | Description                                             |
        +============================+=========================================================+
        | Loss/Loss_cost_critic      | Loss of the cost critic.                                |
        +----------------------------+---------------------------------------------------------+
        | Qc/data_Qc                 | Average cost Q value of offline data.                   |
        +----------------------------+---------------------------------------------------------+
        | Qc/target_Qc               | Average cost Q value of next_obs and next_action.       |
        +----------------------------+---------------------------------------------------------+
        | Qc/current_Qc              | Average cost Q value of obs and agent predicted action. |
        +----------------------------+---------------------------------------------------------+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier.                                |
        +----------------------------+---------------------------------------------------------+
        """
        super()._init_log()

        self._logger.register_key('Loss/Loss_cost_critic')
        self._logger.register_key('Qc/data_Qc')
        self._logger.register_key('Qc/target_Qc')
        self._logger.register_key('Qc/current_Qc')
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _init_model(self) -> None:
        super()._init_model()

        self._cost_critic = (
            CriticBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
                activation=self._cfgs.model_cfgs.critic.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
                num_critics=2,
            )
            .build_critic('q')
            .to(self._device)
        )
        self._target_cost_critic = deepcopy(self._cost_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate must be a float number.'
        self._cost_critic_optimizer = optim.Adam(
            self._cost_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )

        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, action, reward, cost, next_obs, done = batch

        self._update_reward_critic(obs, action, reward, next_obs, done)
        self._update_cost_critic(obs, action, cost, next_obs, done)
        self._update_actor(obs, action)

        self._polyak_update()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            # sample action form the actor
            next_obs_repeat = torch.repeat_interleave(
                next_obs,
                self._cfgs.algo_cfgs.sampled_action_num,
                dim=0,
            )
            next_action = self._actor.predict(next_obs_repeat)

            # compute the target q
            qc1_target, qc2_target = self._target_cost_critic(next_obs_repeat, next_action)
            qc_target = self._cfgs.algo_cfgs.minimum_weighting * torch.min(
                qc1_target,
                qc2_target,
            ) + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.max(qc1_target, qc2_target)
            qc_target = (
                qc_target.reshape(self._cfgs.algo_cfgs.batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            )
            qc_target = cost + (1 - done) * self._cfgs.algo_cfgs.cost_gamma * qc_target
            qc_target = qc_target.squeeze(1)

        qc1, qc2 = self._cost_critic.forward(obs, action)
        critic_loss = nn.functional.mse_loss(qc1, qc_target) + nn.functional.mse_loss(
            qc2,
            qc_target,
        )
        self._cost_critic_optimizer.zero_grad()
        critic_loss.backward()
        self._cost_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic': critic_loss.item(),
                'Qc/data_Qc': qc1[0].mean().item(),
                'Qc/target_Qc': qc_target[0].mean().item(),
            },
        )

    def _update_actor(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        # update vae
        recon_loss, kl_loss = self._actor.vae.loss(obs, action)
        loss = recon_loss + kl_loss
        self._vae_optimizer.zero_grad()
        loss.backward()
        self._vae_optimizer.step()

        # update actor
        action = self._actor.predict(obs)
        qr_curr = self._reward_critic.forward(obs, action)[0]
        qc_curr = self._cost_critic.forward(obs, action)[0]
        actor_loss = -(qr_curr - self._lagrange.lagrangian_multiplier.item() * qc_curr).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        if (
            self.epoch * self._cfgs.algo_cfgs.steps_per_epoch
            > self._cfgs.algo_cfgs.lagrange_start_step
        ):
            self._lagrange.update_lagrange_multiplier(qc_curr.mean().item())

        self._logger.store(
            **{
                'Qr/current_Qr': qr_curr[0].mean().item(),
                'Qc/current_Qc': qc_curr[0].mean().item(),
                'Loss/Loss_actor': actor_loss.item(),
                'Loss/Loss_vae': loss.item(),
                'Loss/Loss_recon': recon_loss.item(),
                'Loss/Loss_kl': kl_loss.item(),
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )

    def _polyak_update(self) -> None:
        super()._polyak_update()
        for target_param, param in zip(
            self._target_cost_critic.parameters(),
            self._cost_critic.parameters(),
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )
