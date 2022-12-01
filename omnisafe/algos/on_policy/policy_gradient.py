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

import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algos import registry
from omnisafe.algos.common.buffer import Buffer
from omnisafe.algos.common.logger import Logger
from omnisafe.algos.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.algos.models.policy_gradient_base import PolicyGradientBase
from omnisafe.algos.utils import core, distributed_utils
from omnisafe.algos.utils.tools import get_flat_params_from


@registry.register
class PolicyGradient(PolicyGradientBase):
    def __init__(self, env, exp_name, data_dir, seed=0, algo='pg', cfgs=None) -> None:
        # Create Environment
        self.env = env
        self.env_id = env.env_id
        self.cfgs = deepcopy(cfgs)
        self.exp_name = exp_name
        self.data_dir = data_dir
        self.algo = algo
        self.use_cost = cfgs['use_cost']
        self.cost_gamma = cfgs['cost_gamma']
        self.local_steps_per_epoch = cfgs['steps_per_epoch'] // distributed_utils.num_procs()
        self.entropy_coef = cfgs['entropy_coef']
        # Call assertions, Check if some variables are valid to experiment
        self._init_checks()
        # Set up logger and save configuration to disk
        # Get local parameters before logger instance to avoid unnecessary print
        self.params = locals()
        self.params['env_id'] = self.env_id
        self.params.pop('self')
        self.params.pop('env')
        self.logger = Logger(exp_name=self.exp_name, data_dir=self.data_dir, seed=self.cfgs['seed'])
        self.logger.save_config(self.params)
        # Set seed
        seed += 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.env.reset(seed=seed)
        # Setup actor-critic module
        self.ac = ConstraintActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            scale_rewards=self.cfgs['scale_rewards'],
            standardized_obs=self.cfgs['standardized_obs'],
            **self.cfgs['model_cfgs'],
        )
        # Set PyTorch + MPI.
        self._init_mpi()
        # Set up experience buffer
        self.buf = Buffer(
            actor_critic=self.ac,
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=self.local_steps_per_epoch,
            scale_rewards=self.cfgs['scale_rewards'],
            standardized_obs=self.cfgs['standardized_obs'],
            **self.cfgs['buffer_cfgs'],
        )
        # Set up optimizers for policy and value function
        self.pi_optimizer = core.get_optimizer(
            'Adam', module=self.ac.pi, learning_rate=self.cfgs['pi_lr']
        )
        self.vf_optimizer = core.get_optimizer(
            'Adam', module=self.ac.v, learning_rate=self.cfgs['vf_lr']
        )
        if self.cfgs['use_cost']:
            self.cf_optimizer = core.get_optimizer(
                'Adam', module=self.ac.c, learning_rate=self.cfgs['vf_lr']
            )
        # Set up scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()
        # Set up model saving
        what_to_save = {
            'pi': self.ac.pi,
            'obs_oms': self.ac.obs_oms,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # Setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.logger.log('Start with training.')

    def _init_learning_rate_scheduler(self):
        scheduler = None
        if self.cfgs['linear_lr_decay']:
            # Linear anneal
            def lm(epoch):
                return 1 - epoch / self.cfgs['epochs']

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.pi_optimizer, lr_lambda=lm)
        return scheduler

    def _init_mpi(self):
        """
        Initialize MPI specifics
        """
        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            dt = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync params across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.ac)
            self.logger.log(f'Done! (took {time.time()-dt:0.3f} sec.)')

    def _init_checks(self):
        """
        Checking feasible
        """
        # The steps in each process should be integer
        assert self.cfgs['steps_per_epoch'] % distributed_utils.num_procs() == 0
        # Ensure local each local process can experience at least one complete eposide
        assert self.env.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()}) or increase '
            f'batch size {self.steps_per_epoch}.'
        )
        # Ensure vilid number for iteration
        assert self.cfgs['pi_iters'] > 0
        assert self.cfgs['critic_iters'] > 0

    def algorithm_specific_logs(self):
        """
        Use this method to collect log information.
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for CPO, etc
        """
        pass

    def check_distributed_parameters(self):
        """
        Check if parameters are synchronized across all processes.
        """

        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {'Policy': self.ac.pi.net, 'Value': self.ac.v.net}
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
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret):
        """
        computing value loss

        Returns:
            torch.Tensor
        """
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def compute_loss_c(self, obs, ret):
        """
        computing cost loss

        Returns:
            torch.Tensor
        """
        return ((self.ac.c(obs) - ret) ** 2).mean()

    def learn(self):
        """
        This is main function for algorithm update, divided into the following steps:
            (1). self.rollout: collect interactive data from environment
            (2). self.update: perform actor/critic updates
            (3). log epoch/update information for visualization and terminal log print.

        Returns:
            model and environment
        """
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.cfgs['epochs']):
            self.epoch_time = time.time()
            # Update internals of AC
            if self.cfgs['exploration_noise_anneal']:
                self.ac.anneal_exploration(frac=epoch / self.cfgs['epochs'])
            if self.cfgs['buffer_cfgs']['reward_penalty']:
                # Consider reward penalty parameter in reward calculation: r' = r - c
                assert hasattr(self, 'lagrangian_multiplier')
                assert hasattr(self, 'lambda_range_projection')
                self.penalty_param = self.lambda_range_projection(self.lagrangian_multiplier)
            else:
                self.penalty_param = 0.0
            # Collect data from environment
            self.env.set_rollout_cfgs(
                local_steps_per_epoch=self.local_steps_per_epoch,
                penalty_param=self.penalty_param,
                use_cost=self.use_cost,
                cost_gamma=self.cost_gamma,
            )
            self.env.roll_out(
                self.ac,
                self.buf,
                self.logger,
            )
            # Update: actor, critic, running statistics
            self.update()

            # Log and store information
            self.log(epoch)
            # Check if all models own the same parameter values
            if epoch % self.cfgs['check_freq'] == 0:
                self.check_distributed_parameters()
            # Save model to disk
            if (epoch + 1) % self.cfgs['save_freq'] == 0:
                self.logger.torch_save(itr=epoch)

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac

    def log(self, epoch: int):
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.cfgs['steps_per_epoch']
        fps = self.cfgs['steps_per_epoch'] / (time.time() - self.epoch_time)
        # Step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs['linear_lr_decay']:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs['pi_lr']

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('Values/Adv', min_and_max=True)
        if self.cfgs['use_cost']:
            self.logger.log_tabular('Values/C', min_and_max=True)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        if self.cfgs['use_cost']:
            self.logger.log_tabular('Loss/Cost')
            self.logger.log_tabular('Loss/DeltaCost')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('Misc/Seed', self.cfgs['seed'])
        self.logger.log_tabular('PolicyRatio')
        self.logger.log_tabular('LR', current_lr)
        if self.cfgs['scale_rewards']:
            reward_scale_mean = self.ac.ret_oms.mean.item()
            reward_scale_stddev = self.ac.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.cfgs['exploration_noise_anneal']:
            noise_std = np.exp(self.ac.pi.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        # Some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()

    def pre_process_data(self, raw_data: dict):
        """
        Pre-process data, e.g. standardize observations, rescale rewards if
            enabled by arguments.

        Parameters
        ----------
        raw_data
            dictionary holding information obtain from environment interactions

        Returns
        -------
        dict
            holding pre-processed data, i.e. observations and rewards
        """
        data = deepcopy(raw_data)
        # Note: use_reward_scaling is currently applied in Buffer...
        # If self.use_reward_scaling:
        #     rew = self.ac.ret_oms(data['rew'], subtract_mean=False, clip=True)
        #     data['rew'] = rew

        if self.cfgs['standardized_obs']:
            assert 'obs' in data
            obs = data['obs']
            data['obs'] = self.ac.obs_oms(obs, clip=False)
        return data

    def update_running_statistics(self, data):
        """
        Update running statistics, e.g. observation standardization,
        or reward scaling. If MPI is activated: sync across all processes.
        """
        if self.cfgs['standardized_obs']:
            self.ac.obs_oms.update(data['obs'])

        # Apply Implement Reward scaling
        if self.cfgs['scale_rewards']:
            self.ac.ret_oms.update(data['discounted_ret'])

    def update(self):
        """
        Update actor, critic, running statistics
        """
        start_times = time.time()
        raw_data = self.buf.get()
        # Pre-process data: standardize observations, advantage estimation, etc.
        data = self.pre_process_data(raw_data)
        # Update critic using epoch data
        self.update_value_net(data=data)
        # Update cost critic using epoch data
        if self.cfgs['use_cost']:
            self.update_cost_net(data=data)
        # Update actor using epoch data
        self.update_policy_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data) -> None:
        # Get prob. distribution before updates: used to measure KL distance
        with torch.no_grad():
            self.p_dist = self.ac.pi.detach_dist(data['obs'])

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs['pi_iters']):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            # Apply L2 norm
            if self.cfgs['use_max_grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.cfgs['max_grad_norm'])

            # Average grads across MPI processes
            distributed_utils.mpi_avg_grads(self.ac.pi.net)
            self.pi_optimizer.step()

            q_dist = self.ac.pi.dist(data['obs'])
            torch_kl = torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()

            if self.cfgs['kl_early_stopping']:
                # Average KL for consistent early stopping across processes
                if distributed_utils.mpi_avg(torch_kl) > self.cfgs['target_kl']:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(
            **{
                'Loss/Pi': self.loss_pi_before,
                'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
                'Misc/StopIter': i + 1,
                'Values/Adv': data['adv'].numpy(),
                'Entropy': pi_info['ent'],
                'KL': torch_kl,
                'PolicyRatio': pi_info['ratio'],
            }
        )

    def update_value_net(self, data: dict) -> None:
        # Divide whole local epoch data into mini_batches which is mbs size
        mbs = self.local_steps_per_epoch // self.cfgs['num_mini_batches']
        assert mbs >= 16, f'Batch size {mbs}<16'

        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = loss_v.item()

        indices = np.arange(self.local_steps_per_epoch)
        val_losses = []
        for _ in range(self.cfgs['critic_iters']):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data['obs'][mb_indices], ret=data['target_v'][mb_indices]
                )
                loss_v.backward()
                val_losses.append(loss_v.item())
                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        self.logger.store(
            **{
                'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
                'Loss/Value': self.loss_v_before,
            }
        )

    def update_cost_net(self, data: dict) -> None:
        """Some child classes require additional updates,
        e.g. Lagrangian-PPO needs Lagrange multiplier parameter."""
        # Ensure we have some key components
        assert self.cfgs['use_cost']
        assert hasattr(self, 'cf_optimizer')
        assert 'target_c' in data, f'provided keys: {data.keys()}'

        if self.cfgs['use_cost']:
            self.loss_c_before = self.compute_loss_c(data['obs'], data['target_c']).item()

        # Divide whole local epoch data into mini_batches which is mbs size
        mbs = self.local_steps_per_epoch // self.cfgs['num_mini_batches']
        assert mbs >= 16, f'Batch size {mbs}<16'

        indices = np.arange(self.local_steps_per_epoch)
        losses = []

        # Train cost value network
        for _ in range(self.cfgs['critic_iters']):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                # Iterate mini batch times
                end = start + mbs
                mb_indices = indices[start:end]

                self.cf_optimizer.zero_grad()
                loss_c = self.compute_loss_c(
                    obs=data['obs'][mb_indices], ret=data['target_c'][mb_indices]
                )
                loss_c.backward()
                losses.append(loss_c.item())
                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.ac.c)
                self.cf_optimizer.step()

        self.logger.store(
            **{
                'Loss/DeltaCost': np.mean(losses) - self.loss_c_before,
                'Loss/Cost': self.loss_c_before,
            }
        )
