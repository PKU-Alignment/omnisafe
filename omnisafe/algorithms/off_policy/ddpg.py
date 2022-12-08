# Copyright 2022 OmniSafe Team. All Rights Reserved.
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

"""Implementation of the DDPG algorithm."""

import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.common.logger import Logger
from omnisafe.common.replay_buffer import ReplayBuffer
from omnisafe.models.constraint_actor_q_critic import ConstraintQActorCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.tools import get_flat_params_from


@registry.register
class DDPG:
    """DDPG specific functions"""

    def __init__(
        self,
        env,
        cfgs,
        algo='ddpg',
    ):
        """initialize"""
        self.env = env
        self.algo = algo
        self.cfgs = deepcopy(cfgs)

        # Set up for learning and rolling out schedule
        self.steps_per_epoch = cfgs.steps_per_epoch
        self.local_steps_per_epoch = self.steps_per_epoch // 1
        self.epochs = cfgs.epochs
        self.total_steps = self.epochs * self.steps_per_epoch
        self.start_steps = cfgs.start_steps
        self.max_ep_len = cfgs.max_ep_len
        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env.env._max_episode_steps
        self.update_after = cfgs.update_after
        self.update_every = cfgs.update_every
        self.num_test_episodes = cfgs.num_test_episodes
        self.env.set_rollout_cfgs(
            deterministic=False, rand_a=True, ep_steps=self.update_every, max_ep_len=self.max_ep_len
        )
        # Ensure local each local process can experience at least one complete episode
        assert self.env.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()}) or increase '
            f'batch size {self.steps_per_epoch}.'
        )
        # Ensure valid number for iteration
        assert cfgs.update_every > 0
        # Set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(cfgs._asdict())
        # Set seed
        seed = cfgs.seed + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.set_seed(seed=seed)
        # Setup actor-critic module
        self.actor_critic = ConstraintQActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            scale_rewards=cfgs.scale_rewards,
            standardized_obs=cfgs.standardized_obs,
            model_cfgs=cfgs.model_cfgs,
        )
        # Set PyTorch + MPI.
        self._init_mpi()
        # Set up experience buffer
        self.buf = ReplayBuffer(
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            **self.cfgs['replay_buffer_cfgs'],
        )
        # Set up optimizer for policy and value function
        self.pi_optimizer = core.get_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=cfgs.actor_lr
        )
        self.vf_optimizer = core.get_optimizer(
            'Adam', module=self.actor_critic.reward_critic, learning_rate=cfgs.critic_lr
        )
        if cfgs.use_cost:
            self.cf_optimizer = core.get_optimizer(
                'Adam', module=self.actor_critic.cost_critic, learning_rate=cfgs.critic_lr
            )
        # Set up scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()
        # Set up target network for off_policy training
        self._ac_training_setup()
        torch.set_num_threads(10)
        # Set up model saving
        what_to_save = {'pi': self.actor_critic.actor}
        self.logger.setup_torch_saver(what_to_save)
        self.logger.torch_save()

        self.start_time = time.time()
        self.epoch_time = time.time()
        self.logger.log('Start with training.')

    def _init_learning_rate_scheduler(self):
        """initialize learning rate scheduler"""
        scheduler = None
        if self.cfgs.use_linear_lr_decay:
            # Linear anneal
            def linear_anneal(epoch):
                return 1 - epoch / self.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer, lr_lambda=linear_anneal
            )
        return scheduler

    def _init_mpi(self):
        """
        Initialize MPI specifics
        """
        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            start = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync parameters across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def algorithm_specific_logs(self):
        """
        Use this method to collect log information.
        """

    def _ac_training_setup(self):
        self.ac_targ = deepcopy(self.actor_critic)
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for parameter in self.ac_targ.pi.parameters():
            parameter.requires_grad = False
        for parameter in self.ac_targ.v.parameters():
            parameter.requires_grad = False
        for parameter in self.ac_targ.c.parameters():
            parameter.requires_grad = False
        if self.algo in ['sac', 'td3', 'sac_lag', 'td3_lag']:
            # Freeze target networks with respect to optimizer (only update via polyak averaging)
            for parameter in self.ac_targ.v_.parameters():
                parameter.requires_grad = False

    def check_distributed_parameters(self):
        """
        Check if parameters are synchronized across all processes.
        """

        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {'Policy': self.actor_critic.pi.net, 'Value': self.actor_critic.v.net}
            for key, module in modules.items():
                flat_params = get_flat_params_from(module).numpy()
                global_min = distributed_utils.mpi_min(np.sum(flat_params))
                global_max = distributed_utils.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_pi(self, data: dict):
        """
        computing pi/actor loss

        Returns:
            torch.Tensor
        """
        action, _ = self.ac.pi.predict(data['obs'], determinstic=True)
        loss_pi = self.ac.v(data['obs'], action)
        pi_info = {}
        return -loss_pi.mean(), pi_info

    def compute_loss_v(self, data):
        """
        computing value loss

        Returns:
            torch.Tensor
        """
        obs, act, rew, obs2, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs2'],
            data['done'],
        )
        q = self.ac.v(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, _ = self.ac_targ.pi.predict(obs, determinstic=True)
            q_targ = self.ac_targ.v(obs2, act_targ)
            backup = rew + self.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        # Useful info for logging
        q_info = dict(Q1Vals=q.detach().numpy())
        return loss_q, q_info

    def compute_loss_c(self, data):
        """
        computing cost loss

        Returns:
            torch.Tensor
        """
        obs, act, cost, obs2, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs2'],
            data['done'],
        )
        qc = self.ac.c(obs, act)

        # Bellman backup for Q function
        with torch.no_grad():
            a, _ = self.ac_targ.pi.predict(obs2, determinstic=True)
            qc_targ = self.ac_targ.c(obs2, a)
            backup = cost + self.gamma * (1 - done) * qc_targ
        # MSE loss against Bellman backup
        loss_qc = ((qc - backup) ** 2).mean()
        # Useful info for logging
        qc_info = dict(QCosts=qc.detach().numpy())

        return loss_qc, qc_info

    def learn(self):
        """
        This is main function for algorithm update, divided into the following steps:
            (1). rollout: collect interactive data from environment
            (2). update: perform actor/critic updates
            (3). log epoch/update information for visualization and terminal log print.

        Returns:
            model and environment
        """
        start_time = time.time()
        for t in range(0, self.local_steps_per_epoch * self.epochs, self.update_every):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            rand_a = t < self.start_steps
            self.env.set_rollout_cfgs(
                deterministic=False,
                rand_a=rand_a,
                ep_steps=self.update_every,
                max_ep_len=self.max_ep_len,
            )
            self.env.roll_out_off(self.ac, self.buf, self.logger, use_cost=self.cfgs['use_cost'])

            # Update handling
            if t >= self.update_after:
                for _ in range(self.update_every):
                    batch = self.buf.sample_batch()
                    self.update(data=batch)

            # End of epoch handling
            if t % self.steps_per_epoch == 0 and t:
                epoch = t // self.steps_per_epoch
                if self.cfgs['exploration_noise_anneal']:
                    self.ac.anneal_exploration(frac=epoch / self.epochs)
                # if self.cfgs['use_cost']:
                #     if self.use_cost_decay:
                #         self.cost_limit_decay(epoch)

                # Save model to disk
                if (epoch + 1) % self.cfgs['save_freq'] == 0:
                    self.logger.torch_save(itr=epoch)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                times = time.time() - start_time
                # Log info about epoch
                self.log(epoch, times, t)
        return self.ac

    def update(self, data):
        """update"""
        # First run one gradient descent step for Q.
        self.update_value_net(data)
        if self.cfgs['use_cost']:
            self.update_cost_net(data)
            for parameter in self.ac.c.parameters():
                parameter.requires_grad = False

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for parameter in self.ac.v.parameters():
            parameter.requires_grad = False

        # Next run one gradient descent step for pi.
        self.update_policy_net(data)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.v.parameters():
            p.requires_grad = True

        if self.cfgs['use_cost']:
            for p in self.ac.c.parameters():
                p.requires_grad = False

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()

    def polyak_update_target(self):
        """polyak update target network"""
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def update_policy_net(self, data) -> None:
        """update policy network"""
        # Train policy with one steps of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.logger.store(**{'Loss/Pi': loss_pi.item()})

    def update_value_net(self, data: dict) -> None:
        """update value network"""
        # Train value critic with one steps of gradient descent
        self.vf_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_v(data)
        loss_q.backward()
        self.vf_optimizer.step()
        self.logger.store(**{'Loss/Value': loss_q.item(), 'Q1Vals': q_info['Q1Vals']})

    def update_cost_net(self, data):
        """update cost network"""
        # Train cost critic with one steps of gradient descent
        self.cf_optimizer.zero_grad()
        loss_qc, qc_info = self.compute_loss_c(data)
        loss_qc.backward()
        self.cf_optimizer.step()
        self.logger.store(**{'Loss/Cost': loss_qc.item(), 'QCosts': qc_info['QCosts']})

    def test_agent(self):
        """test agent"""
        for j in range(self.num_test_episodes):
            self.env.set_rollout_cfgs(
                deterministic=True,
                rand_a=False,
                ep_steps=self.max_ep_len,
                max_ep_len=self.max_ep_len,
            )
            self.env.roll_out_off(self.ac, self.buf, self.logger, use_cost=self.cfgs['use_cost'])

    def log(self, epoch, times, t):
        """log information"""
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.cfgs['steps_per_epoch']
        fps = self.cfgs['steps_per_epoch'] / (time.time() - self.epoch_time)
        # Step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs['use_linear_lr_decay']:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs['pi_lr']

        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCosts')
        self.logger.log_tabular('Metrics/EpLen')
        self.logger.log_tabular('Test/EpRet')
        self.logger.log_tabular('Test/EpCosts')
        self.logger.log_tabular('Test/EpLen')
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('Q1Vals')
        if self.cfgs['use_cost']:
            self.logger.log_tabular('Values/C', min_and_max=True)
            self.logger.log_tabular('QCosts')
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        if self.cfgs['use_cost']:
            self.logger.log_tabular('Loss/Cost')
        self.logger.log_tabular('Misc/Seed', self.cfgs['seed'])
        self.logger.log_tabular('LR', current_lr)
        if self.cfgs['scale_rewards']:
            reward_scale_mean = self.ac.ret_oms.mean.item()
            reward_scale_stddev = self.ac.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.cfgs['exploration_noise_anneal']:
            noise_std = np.exp(self.ac.pi.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', t)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()
