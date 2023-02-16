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

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from omnisafe.utils.model import set_optimizer
from omnisafe.utils.tools import seed_all


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class PolicyGradient:
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `Policy Gradient <https://proceedings.neurips.cc/paper
        /1999/file/64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize PolicyGradient.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        self._algo = self.__class__.__name__
        self._cfgs = cfgs
        self._seed = cfgs.seed + 1000 * distributed.get_rank()
        seed_all(self._seed)
        self._device = torch.device(self._cfgs.device)

        self._env = OnPolicyAdapter(env_id, cfgs.num_envs, self._seed, cfgs)
        assert self._cfgs.steps_per_epoch % distributed.world_size() == 0, (
            'The number of steps per epoch must be divisible by the number of ' 'processes.'
        )
        self._steps_per_epoch = (
            self._cfgs.steps_per_epoch // distributed.world_size() // cfgs.num_envs
        )

        # set up logger and save configuration to disk
        self._logger = Logger(
            output_dir=cfgs.data_dir,
            exp_name=cfgs.exp_name,
            seed=cfgs.seed,
            use_tensorboard=cfgs.use_tensorboard,
            use_wandb=cfgs.use_wandb,
            config=cfgs,
        )

        self._actor_critic = ConstraintActorCritic(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
        ).to(self._device)
        self._set_mpi()

        self._buf = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=cfgs.buffer_cfgs.gamma,
            lam=cfgs.buffer_cfgs.lam,
            lam_c=cfgs.buffer_cfgs.lam_c,
            advantage_estimator=cfgs.buffer_cfgs.adv_estimation_method,
            standardized_adv_r=cfgs.buffer_cfgs.standardized_rew_adv,
            standardized_adv_c=cfgs.buffer_cfgs.standardized_cost_adv,
            penalty_coefficient=cfgs.penalty_param,
            num_envs=cfgs.env_cfgs.num_envs,
            device=self._device,
        )

        # set up optimizer for policy and value function
        self._actor_optimizer = set_optimizer(
            'Adam', module=self._actor_critic.actor, learning_rate=cfgs.actor_lr
        )
        self._reward_critic_optimizer = set_optimizer(
            'Adam', module=self._actor_critic.reward_critic, learning_rate=cfgs.critic_lr
        )
        if cfgs.use_cost:
            self._cost_critic_optimizer = set_optimizer(
                'Adam', module=self._actor_critic.cost_critic, learning_rate=cfgs.critic_lr
            )

        what_to_save = {
            'pi': self._actor_critic.actor,
        }
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._init_log()

    def _set_mpi(self) -> None:
        """Initialize MPI specifics.

        Sync parameters of actor and critic across cores,
        only once necessary."""
        if distributed.world_size() > 1:
            # avoid slowdowns from PyTorch + MPI combo
            distributed.setup_distributed()
            start = time.time()
            self._logger.log('INFO: Sync actor critic parameters')
            # sync parameters across cores: only once necessary, grads are averaged!
            distributed.sync_params(self._actor_critic)
            self._logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def _init_log(self):
        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/reward')

        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio')
        self._logger.register_key('Train/LR')

        self._specific_init_logs()

        # some sub-classes may add information to logs
        self._logger.register_key('TotalEnvSteps')
        self._logger.register_key('Time')
        self._logger.register_key('FPS')

    def _specific_init_logs(self):
        pass

    def learn(self):
        """This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.epochs):
            epoch_time = time.time()
            if self._cfgs.exploration_noise_anneal:
                self._actor_critic.anneal_exploration(frac=epoch / self._cfgs.epochs)

            self._env.roll_out(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )

            self._update()

            self._logger.store(
                **{
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.steps_per_epoch,
                    'FPS': self._cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_optimizer.param_groups[0]['lr'],
                    'Time': (time.time() - start_time),
                }
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.save_freq == 0:
                self._logger.torch_save()

        self._logger.close()

    def _update(self):
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.batch_size,
            shuffle=True,
        )

        for i in range(self._cfgs.actor_iters):
            for (
                obs,
                act,
                target_value_r,
                target_value_c,
                logp,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_rewrad_critic(obs, target_value_r)
                if self._cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_dist = self._actor_critic.actor(original_obs)

            kl = torch.distributions.kl.kl_divergence(old_distribution, new_dist).mean().item()
            kl = distributed.dist_avg(kl)

            if self._cfgs.kl_early_stopping and kl > self._cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Train/StopIter': i + 1,
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': kl,
            }
        )

    def _update_rewrad_critic(self, obs, target_value_r):
        self._reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs), target_value_r)

        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff

        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._reward_critic_optimizer.step()

        self._logger.store(**{'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs, target_value_c):
        self._cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs), target_value_c)

        if self._cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.critic_norm_coeff

        loss.backward()

        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._cost_critic_optimizer.step()

        self._logger.store(**{'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(self, obs, act, logp, adv_r, adv_c):  # pylint: disable=unused-argument
        loss, info = self._loss_pi(obs, act, logp, adv_r)
        self._actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(), self._cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_optimizer.step()
        self._logger.store(
            **{
                'Train/Entropy': info['ent'],
                'Train/PolicyRatio': info['ratio'],
                'Loss/Loss_pi': loss.mean().item(),
            }
        )

    def _loss_pi(self, obs, act, logp, adv) -> Tuple[torch.Tensor, Dict]:
        distribution, logp_ = self._actor_critic.actor(obs, act)
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        approx_kl = (0.5 * (distribution.mean - act) ** 2 / distribution.stddev**2).mean().item()
        entrophy = distribution.entropy().mean().item()
        info = {'kl': approx_kl, 'ent': entrophy, 'ratio': ratio.mean().item()}
        return loss, info
