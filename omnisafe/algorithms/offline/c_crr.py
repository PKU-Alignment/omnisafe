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
"""Implementation of C_CRR."""

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.crr import CRR
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.critic.critic_builder import CriticBuilder


@registry.register
class CCRR(CRR):
    """Constraint variant of CRR.

    References:
        - Title: COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation
        - Author: Lee, JongminPaduraru, CosminMankowitz, Daniel JHeess, NicolasPrecup, Doina
        - URL: `https://arxiv.org/abs/2204.08957`
    """

    def _init_log(self) -> None:
        """Log the C-CRR specific information.

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
            next_action = self._actor.predict(next_obs, deterministic=False)
            qr1_target, qr2_target = self._target_reward_critic(next_obs, next_action)
            qr_target = torch.min(qr1_target, qr2_target)
            qr_target = cost + (1 - done) * self._cfgs.algo_cfgs.gamma * qr_target.unsqueeze(1)
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
                'Loss/Loss_cost_critic': critic_loss.item(),
                'Qc/data_Qc': qr1[0].mean().item(),
                'Qc/target_Qc': qr_target[0].mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-locals
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        qr1, qr2 = self._reward_critic.forward(obs, action)
        qr_data = torch.min(qr1, qr2)

        qc1, qc2 = self._cost_critic.forward(obs, action)
        qc_data = torch.min(qc1, qc2)

        obs_repeat = (
            obs.unsqueeze(1)
            .repeat(1, self._cfgs.algo_cfgs.sampled_action_num, 1)
            .view(obs.shape[0] * self._cfgs.algo_cfgs.sampled_action_num, obs.shape[1])
        )
        act_sample = self._actor.predict(obs_repeat, deterministic=False)

        qr1_sample, qr2_sample = self._reward_critic.forward(obs_repeat, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)
        mean_qr = torch.vstack(
            [q.mean() for q in qr_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_r = qr_data - mean_qr.squeeze(1)

        qc1_sample, qc2_sample = self._reward_critic.forward(obs_repeat, act_sample)
        qc_sample = torch.min(qc1_sample, qc2_sample)
        mean_qc = torch.vstack(
            [q.mean() for q in qc_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_c = qc_data - mean_qc.squeeze(1)

        exp_adv = torch.exp(
            (adv_r - self._lagrange.lagrangian_multiplier.item() * adv_c).detach()
            / self._cfgs.algo_cfgs.beta,
        )
        exp_adv = torch.clamp(exp_adv, 0, 1e10)

        self._actor(obs)
        logp = self._actor.log_prob(action)
        bc_loss = -logp
        policy_loss = (exp_adv * bc_loss).mean()
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

        if (
            self.epoch * self._cfgs.algo_cfgs.steps_per_epoch
            > self._cfgs.algo_cfgs.lagrange_start_step
        ):
            self._lagrange.update_lagrange_multiplier(mean_qc.mean().item())

        self._logger.store(
            **{
                'Loss/Loss_actor': policy_loss.item(),
                'Qr/current_Qr': qr_data[0].mean().item(),
                'Qc/current_Qc': qc_data[0].mean().item(),
                'Train/PolicyStd': self._actor.std,
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
