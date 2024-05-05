# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Safe Stochastic Latent Actor-Critic algorithm."""


from __future__ import annotations

import time

import torch
from rich.progress import track
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter.offpolicy_latent_adapter import OffPolicyLatentAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.common.buffer import OffPolicySequenceBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.latent import CostLatentModel
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SafeSLAC(SACLag):
    """Safe SLAC algorithms for vision-based safe RL tasks.

    References:
        - Title: Safe Reinforcement Learning From Pixels Using a Stochastic Latent Representation.
        - Authors: Yannick Hogewind, Thiago D. Simão, Tal Kachman, Nils Jansen.
        - URL: `Safe SLAC <https://openreview.net/pdf?id=b39dQt_uffW>`_
    """

    _is_latent_model_init_learned: bool

    def _init(self) -> None:
        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

            assert self._cfgs.model_cfgs.critic.lr is not None
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha],
                lr=self._cfgs.model_cfgs.critic.lr,
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device),
            )

        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self._buf: OffPolicySequenceBuffer = OffPolicySequenceBuffer(  # type: ignore
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            device=self._device,
            num_sequences=self._cfgs.algo_cfgs.num_sequences,
        )
        self._is_latent_model_init_learned = False

    def _init_env(self) -> None:
        self._env: OffPolicyLatentAdapter = OffPolicyLatentAdapter(  # type: ignore
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
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
        self._update_latent_count = 0

    def _init_model(self) -> None:
        self._cfgs.model_cfgs.critic['num_critics'] = 2

        assert self._env.observation_space.shape
        assert self._env.action_space.shape

        self._latent_model = CostLatentModel(
            obs_shape=self._env.observation_space.shape,
            act_shape=self._env.action_space.shape,
            feature_dim=self._cfgs.algo_cfgs.feature_dim,
            latent_dim_1=self._cfgs.algo_cfgs.latent_dim_1,
            latent_dim_2=self._cfgs.algo_cfgs.latent_dim_2,
            hidden_sizes=self._cfgs.algo_cfgs.hidden_sizes,
            image_noise=self._cfgs.algo_cfgs.image_noise,
        ).to(self._device)
        self._update_latent_count = 0

        self._actor_critic: ConstraintActorQCritic = ConstraintActorQCritic(
            obs_space=self._env.latent_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        self._latent_model_optimizer = optim.Adam(
            self._latent_model.parameters(),
            lr=1e-4,
        )

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch + 1,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    latent_model=self._latent_model,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _prepare_batch(self, obs_: torch.Tensor, action_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            feature_ = self._latent_model.encoder(obs_)
            z_ = torch.cat(self._latent_model.sample_posterior(feature_, action_)[2:4], dim=-1)

        z, next_z = z_[:, -2], z_[:, -1]
        action = action_[:, -1]

        return z, next_z, action

    def _update(self) -> None:
        if not self._is_latent_model_init_learned:
            for _ in track(
                range(self._cfgs.algo_cfgs.latent_model_init_learning_steps),
                description='initial updating of latent model...',
            ):
                self._update_latent_model()
            self._is_latent_model_init_learned = True

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            self._update_latent_model()

            data = self._buf.sample_batch(64)
            self._update_count += 1
            obs_, act_, reward, cost, done = (
                data['obs'],
                data['act'],
                data['reward'][:, -1].squeeze(),
                data['cost'][:, -1].squeeze(),
                data['done'][:, -1].squeeze(),
            )
            obs, next_obs, act = self._prepare_batch(obs_, act_)
            self._update_reward_critic(obs, act, reward, done, next_obs)
            self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_latent_model(self) -> None:
        data = self._buf.sample_batch(32)
        obs, act, reward, cost, done = (
            data['obs'],
            data['act'],
            data['reward'],
            data['cost'],
            data['done'],
        )

        self._update_latent_count += 1
        loss_kld, loss_image, loss_reward, loss_cost = self._latent_model.calculate_loss(
            obs,
            act,
            reward,
            done,
            cost,
        )

        self._latent_model_optimizer.zero_grad()
        (loss_kld + loss_image + loss_reward + loss_cost).backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._latent_model.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._latent_model_optimizer.step()
