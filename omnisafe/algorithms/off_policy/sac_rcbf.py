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
"""Implementation of the Soft Actor-Critic algorithm with Robust Control Barrier Function."""
# mypy: ignore-errors

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter.robust_barrier_function_adapter import RobustBarrierFunctionAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.robust_barrier_solver import CBFQPLayer
from omnisafe.common.robust_gp_model import DynamicsModel
from omnisafe.utils.distributed import get_rank


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACRCBF(SAC):
    """The Soft Actor-Critic algorithm with Robust Control Barrier Function.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init_env(self) -> None:
        self._env: RobustBarrierFunctionAdapter = RobustBarrierFunctionAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        solver = CBFQPLayer(
            env=self._env,
            device=self._cfgs.train_cfgs.device,
            gamma_b=self._cfgs.cbf_cfgs.gamma_b,
            k_d=self._cfgs.cbf_cfgs.k_d,
            l_p=self._cfgs.cbf_cfgs.l_p,
        )
        dynamics_model = DynamicsModel(env=self._env)

        self._env.set_dynamics_model(dynamics_model=dynamics_model)
        self._env.set_solver(solver=solver)

        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        super()._update_actor(obs)

        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                action = self._env.get_safe_action(obs, action)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                {
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                },
            )
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_action = self._env.get_safe_action(next_obs, next_action)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
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
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        action = self._env.get_safe_action(obs, action)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()

    def _specific_save(self) -> None:
        """Save some algorithms specific models per epoch."""
        super()._specific_save()
        if get_rank() == 0:
            path = os.path.join(self._logger.log_dir, 'gp_model_save')
            os.makedirs(path, exist_ok=True)
            train_x = self._env.dynamics_model.train_x
            train_y = self._env.dynamics_model.train_y
            disturb_estimators = self._env.dynamics_model.disturb_estimators
            weights = []
            for disturb_estimator in disturb_estimators:
                weights.append(disturb_estimator.model.state_dict())
            torch.save(weights, os.path.join(path, f'gp_models_{self._logger.current_epoch}.pkl'))
            torch.save(
                train_x,
                os.path.join(path, f'gp_models_train_x_{self._logger.current_epoch}.pkl'),
            )
            torch.save(
                train_y,
                os.path.join(path, f'gp_models_train_y_{self._logger.current_epoch}.pkl'),
            )
