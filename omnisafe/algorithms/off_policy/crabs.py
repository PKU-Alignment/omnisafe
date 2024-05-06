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
"""Implementation of the Co-trained Barrier Certificate for Safe RL algorithm."""
# pylint: disable=all
import time
from copy import deepcopy
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from rich.progress import track
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter.crabs_adapter import CRABSAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.control_barrier_function.crabs.models import (
    AddGaussianNoise,
    CrabsCore,
    ExplorationPolicy,
    MeanPolicy,
    MultiLayerPerceptron,
    UniformPolicy,
)
from omnisafe.common.control_barrier_function.crabs.optimizers import (
    Barrier,
    BarrierCertOptimizer,
    PolicyAdvTraining,
    SLangevinOptimizer,
    StateBox,
)
from omnisafe.common.control_barrier_function.crabs.utils import (
    Normalizer,
    create_model_and_trainer,
    get_pretrained_model,
)
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class CRABS(SAC):
    """The CRABS algorithm.

    References:
        - Title: Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations
        - Authors: Yuping Luo, Tengyu Ma.
        - URL: `CRABS <https://arxiv.org/abs/2108.01846>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.CRABSAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: CRABSAdapter = CRABSAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        self._epochs: int = self._cfgs.train_cfgs.num_epochs
        self._epoch: int = 0
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Initialize the models.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        super()._init_model()
        self.s0 = torch.tensor(
            self._env.reset()[0],
            device=self._cfgs.train_cfgs.device,
            dtype=torch.float32,
        )
        self.dim_state = self._env.observation_space.shape[0]  # type: ignore
        self.dim_action = self._env.action_space.shape[0]  # type: ignore

        self.normalizer = Normalizer(self.dim_state, clip=1000).to(self._device)
        self.state_box = StateBox([self.dim_state], self.s0, self._device, logger=None)
        self.state_box.reset()

        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        self.mean_policy = MeanPolicy(self._actor_critic.actor)

        self.model, self.model_trainer = create_model_and_trainer(
            self._cfgs,
            self.dim_state,
            self.dim_action,
            self.normalizer,
            self._device,
        )

    def _init_log(self) -> None:
        super()._init_log()
        what_to_save: dict[str, Any] = {}  # type: ignore
        what_to_save['pi'] = self._actor_critic.actor
        what_to_save['h'] = self.h
        what_to_save['models'] = self.model
        what_to_save['obs_normalizer'] = self.normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()
        self._logger.register_key(
            'Metrics/RawPolicyEpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/RawPolicyEpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/RawPolicyEpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()

        In CRABS, we need to initialize the ``barrier function``, ``world models``, ``policy`` and ``optimizers``.
        """
        super()._init()

        self.h = Barrier(
            nn.Sequential(self.normalizer, MultiLayerPerceptron([self.dim_state, 256, 256, 1])),
            self._env._env.env.barrier_fn,
            self.s0,
            self._cfgs.lyapunov,
        ).to(self._device)

        self.core = CrabsCore(self.h, self.model, self.mean_policy, self._cfgs.crabs)  # type: ignore
        self.barrier = self.core.u
        if self._cfgs.transition_model_cfgs.frozen:
            self.model.requires_grad_(False)
            self._logger.log('Warning: models are frozen!')

        self.load_from_ckpt()

        self.core_ref = deepcopy(self.core)

        self.state_box.find_box(self.core_ref.h)

        self.s_opt = SLangevinOptimizer(
            self.core,
            self.state_box,
            self._cfgs.train_cfgs.device,
            self._cfgs.opt_s,
            logger=None,
        ).to(
            self._device,
        )

        self.n_samples_so_far = 0

        self.h_opt = BarrierCertOptimizer(
            self.h,
            self.core.obj_eval,
            self.core_ref,
            self.s_opt,
            self.state_box,
            cfgs=self._cfgs,
        )
        self.policy_adv_opt = PolicyAdvTraining(
            self._actor_critic.actor,
            self.s_opt,
            self.core.obj_eval,
            self._cfgs,
        )

    def learn(self):
        """This is the main learning function of CRABS, orchestrating the training process across multiple epochs.

        It performs the following operations sequentially:
        1. Pretraining steps on s*.
        2. Rollout and evaluation procedures using different policies.
        3. Training the world models using the collected data.
        4. Optimizing s*.
        5. Training policy using SAC.
        6. Training barrier function.

        Returns:
            ep_ret: Average episode return in the final epoch.
            ep_cost: Average episode cost in the final epoch.
            ep_len: Average episode length in the final epoch.
        """
        self.h_opt.h_ref = self.core_ref.h
        self._logger.log('Info: pretrain s...')
        for i in range(self._cfgs.n_pretrain_s_iters):
            if i % 1000 == 0:
                self.s_opt.debug(step=i)
            self.s_opt.step()

        self._env.rollout(
            rollout_step=self._env.horizon * 10,
            agent=ExplorationPolicy(self._actor_critic.actor, self.core_ref),  # type: ignore
            buffer=self._buf,
            logger=self._logger,
            use_rand_action=False,
        )

        self.train_models(epochs=5)

        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0

        freq = 0.5

        for epoch in range(self._epochs):
            self._epoch = epoch

            rollout_time = 0.0
            update_time = 0.0
            eval_time = 0.0
            epoch_time = time.time()

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.raw_policy_episodes,
                agent=self.mean_policy,
                logger=self._logger,
            )

            eval_time += time.time() - eval_start

            rollout_start = time.time()

            self._env.rollout(
                rollout_step=self._env.horizon * 2,
                agent=ExplorationPolicy(
                    AddGaussianNoise(
                        self._actor_critic.actor,  # type: ignore
                        0.0,
                        self._cfgs.algo_cfgs.exploration_noise,
                    ),
                    self.core_ref,
                ),
                buffer=self._buf,
                logger=self._logger,
                use_rand_action=False,
            )

            step += self._env.horizon * 2

            self._env.rollout(
                rollout_step=self._env.horizon * 2,
                agent=ExplorationPolicy(UniformPolicy(self.dim_action), self.core_ref),  # type: ignore
                buffer=self._buf,
                logger=self._logger,
                use_rand_action=False,
            )

            step += self._env.horizon * 2
            rollout_time += time.time() - rollout_start

            self.train_models(epochs=1)

            self._logger.log(f'Info: Epoch {epoch}: train policy, safety req freq = {freq:.3f}')

            for _ in track(range(2000), description='Optimizing s...'):
                self.s_opt.step()

            for t in track(range(2000), description='Updating Policy...'):
                if t % 1000 == 0:
                    eval_start = time.time()
                    self._env.eval_policy(
                        episode=self._cfgs.train_cfgs.raw_policy_episodes,
                        agent=self.mean_policy,
                        logger=self._logger,
                    )
                    eval_time += time.time() - eval_start

                    rollout_start = time.time()
                    self._env.rollout(
                        rollout_step=self._env.horizon,
                        agent=ExplorationPolicy(self._actor_critic.actor, self.core_ref),  # type: ignore
                        buffer=self._buf,
                        logger=self._logger,
                        use_rand_action=False,
                    )
                    step += self._env.horizon
                    rollout_time += time.time() - rollout_start

                if len(self._buf) > 1000:
                    update_start = time.time()
                    self._update()  # optimize unsafe policy
                    update_time += time.time() - update_start
                update_start = time.time()
                self.policy_adv_opt.step(freq)
                update_time += time.time() - update_start

            self._logger.log('Info: train h!')
            update_start = time.time()
            found, multiplier = self.h_opt.train()
            update_time += time.time() - update_start
            freq = np.clip(freq * multiplier, 0.1, 10)
            if found:
                self._logger.log(f'Info: reduce frequency to {freq:.3f}. Reset core_ref')
                self.core_ref.load_state_dict(self.core.state_dict())
            else:
                self._logger.log(f"Warning: can't find h, increase freq to {freq}")
                self.h.load_state_dict(self.core_ref.h.state_dict())
            self.h_opt.check_by_grad()

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
                    'TotalEnvSteps': step + 1,
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

    def get_masked_q(self, min_q_value, states, actions):
        """Get the masked Q value.

        The loss function here is based on SAC, but with an additional term to mask the Q value.

        Args:
            min_q_value (torch.Tensor): The minimum Q value.
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.

        Returns:
            torch.Tensor: The masked Q value.
        """
        barrier = self.barrier(states, actions)
        return torch.where(
            barrier > 0,
            torch.full([len(barrier)], -1000.0, device=self._cfgs.train_cfgs.device) - barrier,
            min_q_value,
        )

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        The loss function here is based on SAC, but with an additional term to mask the Q value
        when the barrier function is positive.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value = self.get_masked_q(
                torch.min(next_q1_value_r, next_q2_value_r),
                next_obs,
                next_action,
            )
            valid_mask = self.barrier(next_obs, next_action) <= 0
            next_q_value_r = next_q_value - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(
            q1_value_r * valid_mask,
            target_q_value_r * valid_mask,
        ) + nn.functional.mse_loss(
            q2_value_r * valid_mask,
            target_q_value_r * valid_mask,
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
        r"""Computing ``pi/actor`` loss.

        The loss function here is based on SAC, but with an additional term to mask the Q value
        when the barrier function is positive.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        q_value = self.get_masked_q(torch.min(q1_value_r, q2_value_r), obs, action)
        return (self._alpha * log_prob - q_value).mean()

    def load_from_ckpt(self):
        """Load pretrained model from url.

        If the model is not found locally, download it from the cloud.
        """
        model_url = (
            'https://drive.google.com/uc?export=download&id=1MciBSHU74HjADTINUMrlpa7s_9cX7f1P'
        )
        model_path = '~/.cache/omnisafe/pretrain_models/crabs_swing_models_checkpoint.pt'
        param_dict = get_pretrained_model(model_path, model_url, self._device)
        self._actor_critic.actor.load_state_dict(param_dict['actor'])
        print('Load policy')
        self.h.load_state_dict(param_dict['h'])
        print('Load h')
        self.model.load_state_dict(param_dict['model'])
        print('Load transition model')

    def train_models(self, *, epochs):
        """Train the transition models.

        Args:
            epochs (int): The number of epochs to train the model.
        """
        from torch.utils.data import DataLoader, IterableDataset

        class BufferDataset(IterableDataset):
            def __init__(self, buffer, batch_size, n_iters_per_epoch=None) -> None:
                self.buffer = buffer
                self.batch_size = batch_size
                self.n_iters_per_epoch = n_iters_per_epoch

            def __iter__(self):
                n_iters = (
                    self.n_iters_per_epoch
                    if self.n_iters_per_epoch is not None
                    else len(self.buffer) // self.batch_size
                )
                for _ in range(n_iters):
                    batch = self.buffer.sample_batch(batch_size=self.batch_size)
                    yield batch

        buffer_dataset = BufferDataset(self._buf, 256, n_iters_per_epoch=1000)
        train_dataloader = DataLoader(buffer_dataset, batch_size=None)

        self.model_trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices=[int(str(self._device)[-1])],
            default_root_dir=self._cfgs.logger_cfgs.log_dir,
        )
        self.model_trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
        )
        self.model.to(self._device)
