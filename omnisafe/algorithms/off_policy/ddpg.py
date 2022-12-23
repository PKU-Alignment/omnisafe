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
from omnisafe.common.base_buffer import BaseBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import create_dict_from_namedtuple
from omnisafe.utils.tools import get_flat_params_from
from omnisafe.wrappers import wrapper_registry


@registry.register
class DDPG:  # pylint: disable=too-many-instance-attributes
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        Title: Continuous control with deep reinforcement learning
        Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez,
                 Yuval Tassa, David Silver, Daan Wierstra.
        URL: https://arxiv.org/abs/1509.02971
    """

    def __init__(
        self,
        env_id: str,
        cfgs=None,
    ):
        """Initialize DDPG.

        Args:
            env_id (str): Environment ID.
            cfgs (dict): Configuration dictionary.
            algo (str): Algorithm name.
            wrapper_type (str): Wrapper type.
        """
        self.cfgs = deepcopy(cfgs)
        self.wrapper_type = self.cfgs.wrapper_type
        self.env = wrapper_registry.get(self.wrapper_type)(
            env_id,
            use_cost=cfgs.use_cost,
            max_ep_len=cfgs.max_ep_len,
        )
        self.env_id = env_id
        self.algo = self.__class__.__name__

        # Set up for learning and rolling out schedule
        self.steps_per_epoch = cfgs.steps_per_epoch
        self.local_steps_per_epoch = cfgs.steps_per_epoch
        self.epochs = cfgs.epochs
        self.total_steps = self.epochs * self.steps_per_epoch
        self.start_steps = cfgs.start_steps
        # The steps in each process should be integer
        assert cfgs.steps_per_epoch % distributed_utils.num_procs() == 0
        # Ensure local each local process can experience at least one complete episode
        assert self.env.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()}) or increase '
            f'batch size {self.steps_per_epoch}.'
        )
        # Ensure valid number for iteration
        assert cfgs.update_every > 0
        self.max_ep_len = cfgs.max_ep_len
        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env.env._max_episode_steps
        self.update_after = cfgs.update_after
        self.update_every = cfgs.update_every
        self.num_test_episodes = cfgs.num_test_episodes

        self.env.set_rollout_cfgs(
            determinstic=False,
            rand_a=True,
        )

        # Set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(create_dict_from_namedtuple(cfgs))
        # Set seed
        seed = cfgs.seed + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.set_seed(seed=seed)
        # Setup actor-critic module
        self.actor_critic = ConstraintActorQCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            standardized_obs=cfgs.standardized_obs,
            model_cfgs=cfgs.model_cfgs,
        )
        # Set PyTorch + MPI.
        self._init_mpi()
        # Set up experience buffer
        # obs_dim, act_dim, size, batch_size
        self.buf = BaseBuffer(
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=cfgs.replay_buffer_cfgs.size,
            batch_size=cfgs.replay_buffer_cfgs.batch_size,
        )
        # Set up optimizer for policy and q-function
        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=cfgs.actor_lr
        )
        self.critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.critic, learning_rate=cfgs.critic_lr
        )
        if cfgs.use_cost:
            self.cost_critic_optimizer = core.set_optimizer(
                'Adam', module=self.actor_critic.cost_critic, learning_rate=cfgs.critic_lr
            )
        # Set up scheduler for policy learning rate decay
        self.scheduler = self.set_learning_rate_scheduler()
        # Set up target network for off_policy training
        self._ac_training_setup()
        torch.set_num_threads(10)
        # Set up model saving
        what_to_save = {
            'pi': self.actor_critic.actor,
            'obs_oms': self.actor_critic.obs_oms,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # Set up timer
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.logger.log('Start with training.')

    def set_learning_rate_scheduler(self):
        """Set up learning rate scheduler."""

        scheduler = None
        if self.cfgs.linear_lr_decay:
            # Linear anneal
            def linear_anneal(epoch):
                return 1 - epoch / self.cfgs.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.actor_optimizer, lr_lambda=linear_anneal
            )
        return scheduler

    def _init_mpi(self):
        """Initialize MPI specifics."""

        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            start = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync params across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def algorithm_specific_logs(self):
        """Use this method to collect log information."""

    def _ac_training_setup(self):
        """Set up target network for off_policy training."""
        self.ac_targ = deepcopy(self.actor_critic)
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self.ac_targ.actor.parameters():
            param.requires_grad = False
        for param in self.ac_targ.critic.parameters():
            param.requires_grad = False
        for param in self.ac_targ.cost_critic.parameters():
            param.requires_grad = False

    def check_distributed_parameters(self):
        """Check if parameters are synchronized across all processes."""
        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {'Policy': self.actor_critic.actor.net, 'Value': self.actor_critic.critic.net}
            for key, module in modules.items():
                flat_params = get_flat_params_from(module).numpy()
                global_min = distributed_utils.mpi_min(np.sum(flat_params))
                global_max = distributed_utils.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Args:
            data (dict): data dictionary.

        Returns:
            torch.Tensor.
        """
        action, _ = self.actor_critic.actor.predict(data['obs'], deterministic=True)
        loss_pi = self.actor_critic.critic(data['obs'], action)[0]
        pi_info = {}
        return -loss_pi.mean(), pi_info

    def compute_loss_v(self, data):
        """Computing value loss.

        Args:
            data (dict): data dictionary.

        Returns:
            torch.Tensor.
        """
        obs, act, rew, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
        q_value = self.actor_critic.critic(obs, act)[0]
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ = self.ac_targ.actor.predict(obs, deterministic=True, need_log_prob=False)
            q_targ = self.ac_targ.critic(obs_next, act_targ)[0]
            backup = rew + self.cfgs.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q_value - backup) ** 2).mean()
        # Useful info for logging
        q_info = dict(QVals=q_value.detach().numpy())
        return loss_q, q_info

    def compute_loss_c(self, data):
        """Computing cost loss.

        Args:
            data (dict): data dictionary.

        Returns:
            torch.Tensor.
        """
        obs, act, cost, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
        cost_q_value = self.actor_critic.cost_critic(obs, act)[0]

        # Bellman backup for Q function
        with torch.no_grad():
            action, _ = self.ac_targ.actor.predict(obs_next, deterministic=True)
            qc_targ = self.ac_targ.cost_critic(obs_next, action)[0]
            backup = cost + self.cfgs.gamma * (1 - done) * qc_targ
        # MSE loss against Bellman backup
        loss_qc = ((cost_q_value - backup) ** 2).mean()
        # Useful info for logging
        qc_info = dict(QCosts=cost_q_value.detach().numpy())

        return loss_qc, qc_info

    def learn(self):
        """
        This is main function for algorithm update, divided into the following steps:
            (1). self.rollout: collect interactive data from environment
            (2). self.update: perform actor/critic updates
            (3). log epoch/update information for visualization and terminal log print.

        Returns:
            model and environment.
        """

        for steps in range(0, self.local_steps_per_epoch * self.epochs, self.update_every):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            use_rand_action = steps < self.start_steps
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=False,
                use_rand_action=use_rand_action,
                ep_steps=self.update_every,
            )

            # Update handling
            if steps >= self.update_after:
                for _ in range(self.update_every):
                    batch = self.buf.sample_batch()
                    self.update(data=batch)

            # End of epoch handling
            if steps % self.steps_per_epoch == 0 and steps:
                epoch = steps // self.steps_per_epoch
                if self.cfgs.exploration_noise_anneal:
                    self.actor_critic.anneal_exploration(frac=epoch / self.epochs)
                # if self.cfgs.use_cost_critic:
                #     if self.use_cost_decay:
                #         self.cost_limit_decay(epoch)

                # Save model to disk
                if (epoch + 1) % self.cfgs.save_freq == 0:
                    self.logger.torch_save(itr=epoch)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                # Log info about epoch
                self.log(epoch, steps)
        return self.actor_critic

    def update(self, data):
        """Update.

        Args:
            data (dict): data dictionary.
        """
        # First run one gradient descent step for Q.
        self.update_value_net(data)
        if self.cfgs.use_cost:
            self.update_cost_net(data)
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = False

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for actor.
        self.update_policy_net(data)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()

    def polyak_update_target(self):
        """Polyak update target network."""
        with torch.no_grad():
            for param, param_targ in zip(self.actor_critic.parameters(), self.ac_targ.parameters()):
                # Notes: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                param_targ.data.mul_(self.cfgs.polyak)
                param_targ.data.add_((1 - self.cfgs.polyak) * param.data)

    def update_policy_net(self, data) -> None:
        """Update policy network.

        Args:
            data (dict): data dictionary.
        """
        # Train policy with one steps of gradient descent
        self.actor_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()
        self.logger.store(**{'Loss/Pi': loss_pi.item()})

    def update_value_net(self, data: dict) -> None:
        """Update value network.

        Args:
            data (dict): data dictionary
        """
        # Train value critic with one steps of gradient descent
        self.critic_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_v(data)
        loss_q.backward()
        self.critic_optimizer.step()
        self.logger.store(**{'Loss/Value': loss_q.item(), 'QVals': q_info['QVals']})

    def update_cost_net(self, data):
        """Update cost network.

        Args:
            data (dict): data dictionary.
        """
        # Train cost critic with one steps of gradient descent
        self.cost_critic_optimizer.zero_grad()
        loss_qc, qc_info = self.compute_loss_c(data)
        loss_qc.backward()
        self.cost_critic_optimizer.step()
        self.logger.store(**{'Loss/Cost': loss_qc.item(), 'QCosts': qc_info['QCosts']})

    def test_agent(self):
        """Test agent."""
        for _ in range(self.num_test_episodes):
            # self.env.set_rollout_cfgs(deterministic=True, rand_a=False)
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=True,
                use_rand_action=False,
                ep_steps=self.max_ep_len,
            )

    def log(self, epoch, total_steps):
        """Log info about epoch."""
        fps = self.cfgs.steps_per_epoch / (time.time() - self.epoch_time)
        # Step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs.linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs.actor_lr

        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')
        self.logger.log_tabular('Test/EpRet')
        self.logger.log_tabular('Test/EpCost')
        self.logger.log_tabular('Test/EpLen')
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('QVals')
        if self.cfgs.use_cost:
            self.logger.log_tabular('Values/C', min_and_max=True)
            self.logger.log_tabular('QCosts')
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        if self.cfgs.use_cost:
            self.logger.log_tabular('Loss/Cost')
        self.logger.log_tabular('Misc/Seed', self.cfgs.seed)
        self.logger.log_tabular('LR', current_lr)
        if self.cfgs.scale_rewards:
            reward_scale_mean = self.actor_critic.ret_oms.mean.item()
            reward_scale_stddev = self.actor_critic.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.cfgs.exploration_noise_anneal:
            noise_std = np.exp(self.actor_critic.actor.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()
