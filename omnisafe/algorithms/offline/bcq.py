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
"""Implementation of BCQ."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.models.actor.perturbation_actor import PerturbationActor
from omnisafe.models.critic.critic_builder import CriticBuilder


@registry.register
class BCQ(BaseOffline):
    """Batch-Constrained Deep Reinforcement Learning.

    References:
        - Title: Off-Policy Deep Reinforcement Learning without Exploration
        - Author: Fujimoto, ScottMeger, DavidPrecup, Doina.
        - URL: `https://arxiv.org/abs/1812.02900`
    """

    def _init_log(self) -> None:
        """Log the BCQ specific information.

        +-------------------------+----------------------------------------------------+
        | Things to log           | Description                                        |
        +=========================+====================================================+
        | Loss/Loss_vae           | Loss of VAE network                                |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_recon         | Reconstruction loss of VAE network                 |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_kl            | KL loss of VAE network                             |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_actor         | Loss of the actor network.                         |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                         |
        +-------------------------+----------------------------------------------------+
        | Qr/data_Qr              | Average Q value of offline data.                   |
        +-------------------------+----------------------------------------------------+
        | Qr/target_Qr            | Average Q value of next_obs and next_action.       |
        +-------------------------+----------------------------------------------------+
        | Qr/current_Qr           | Average Q value of obs and agent predicted action. |
        +-------------------------+----------------------------------------------------+
        """
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'actor': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)

        self._logger.register_key('Loss/Loss_vae')
        self._logger.register_key('Loss/Loss_recon')
        self._logger.register_key('Loss/Loss_kl')

        self._logger.register_key('Loss/Loss_reward_critic')
        self._logger.register_key('Loss/Loss_actor')
        self._logger.register_key('Qr/data_Qr')
        self._logger.register_key('Qr/target_Qr')
        self._logger.register_key('Qr/current_Qr')

    def _init_model(self) -> None:
        self._actor: PerturbationActor = PerturbationActor(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
            activation=self._cfgs.model_cfgs.actor.activation,
            weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
        ).to(self._device)
        self._actor.phi = self._cfgs.algo_cfgs.phi
        assert isinstance(
            self._cfgs.model_cfgs.actor.lr,
            float,
        ), 'The learning rate of actor must be a float number.'
        self._actor_optimizer = optim.Adam(
            self._actor.perturbation.parameters(),
            lr=self._cfgs.model_cfgs.actor.lr,
        )
        self._vae_optimizer = optim.Adam(
            self._actor.vae.parameters(),
            lr=self._cfgs.model_cfgs.actor.lr,
        )

        self._reward_critic = (
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
        self._target_reward_critic = deepcopy(self._reward_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate of actor must be a float number.'
        self._reward_critic_optimizer = optim.Adam(
            self._reward_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, action, reward, _, next_obs, done = batch

        self._update_reward_critic(obs, action, reward, next_obs, done)
        self._update_actor(obs, action)

        self._polyak_update()

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            # sample action from the actor
            next_obs_repeat = torch.repeat_interleave(
                next_obs,
                self._cfgs.algo_cfgs.sampled_action_num,
                dim=0,
            )
            next_action = self._actor.predict(next_obs_repeat)

            # compute target q
            qr1_target, qr2_target = self._target_reward_critic(next_obs_repeat, next_action)
            qr_target = self._cfgs.algo_cfgs.minimum_weighting * torch.min(
                qr1_target,
                qr2_target,
            ) + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.max(qr1_target, qr2_target)
            qr_target = (
                qr_target.reshape(self._cfgs.algo_cfgs.batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            )
            qr_target = reward + (1 - done) * self._cfgs.algo_cfgs.gamma * qr_target
            qr_target = qr_target.squeeze(1)

        qr1, qr2 = self._reward_critic.forward(obs, action)
        critic_loss = nn.functional.mse_loss(qr1, qr_target) + nn.functional.mse_loss(
            qr2,
            qr_target,
        )
        self._reward_critic_optimizer.zero_grad()
        critic_loss.backward()
        self._reward_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_reward_critic': critic_loss.item(),
                'Qr/data_Qr': qr1[0].mean().item(),
                'Qr/target_Qr': qr_target[0].mean().item(),
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
        actor_loss = -qr_curr.mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self._logger.store(
            **{
                'Qr/current_Qr': qr_curr[0].mean().item(),
                'Loss/Loss_actor': actor_loss.item(),
                'Loss/Loss_vae': loss.item(),
                'Loss/Loss_recon': recon_loss.item(),
                'Loss/Loss_kl': kl_loss.item(),
            },
        )

    def _polyak_update(self) -> None:
        for target_param, param in zip(
            self._target_reward_critic.parameters(),
            self._reward_critic.parameters(),
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )
