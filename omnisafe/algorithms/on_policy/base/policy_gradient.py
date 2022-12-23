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
"""Implementation of the Policy Gradient algorithm."""

import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.common.buffer import Buffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import create_dict_from_namedtuple
from omnisafe.utils.tools import get_flat_params_from
from omnisafe.wrappers import wrapper_registry


@registry.register
class PolicyGradient:  # pylint: disable=too-many-instance-attributes
    """The Policy Gradient algorithm.

    References:
        Paper Name: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        Paper Author: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
        Paper URL: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

    """

    # pylint: disable-next=too-many-locals
    def __init__(
        self,
        env_id,
        cfgs=None,
    ) -> None:
        """Initialize the algorithm.

        Args:
            env: The environment.
            algo: (default: :const:`PolicyGradient`)
                Name of the algorithm for logging process data.
            cfgs: (default: :const:`None`)
                This is a dictionary of the algorithm hyper-parameters.
        """
        self.algo = self.__class__.__name__
        self.cfgs = deepcopy(cfgs)
        self.wrapper_type = self.cfgs.wrapper_type
        self.env = wrapper_registry.get(self.wrapper_type)(
            env_id, cfgs=self.cfgs._asdict().get('env_cfgs')
        )

        assert self.cfgs.steps_per_epoch % distributed_utils.num_procs() == 0
        self.local_steps_per_epoch = cfgs.steps_per_epoch // distributed_utils.num_procs()

        # Ensure local each local process can experience at least one complete episode
        assert self.env.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()}) or increase '
            f'batch size {self.cfgs.steps_per_epoch}.'
        )

        # Set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(create_dict_from_namedtuple(cfgs))
        # Set seed
        seed = int(cfgs.seed) + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.env.reset(seed=seed)
        # Setup actor-critic module
        self.actor_critic = ConstraintActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            scale_rewards=cfgs.scale_rewards,
            standardized_obs=cfgs.standardized_obs,
            model_cfgs=cfgs.model_cfgs,
        )
        # Set PyTorch + MPI.
        self.set_mpi()
        # Set up experience buffer

        self.buf = Buffer(
            actor_critic=self.actor_critic,
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=self.local_steps_per_epoch,
            reward_penalty=cfgs.reward_penalty,
            scale_rewards=cfgs.scale_rewards,
            standardized_obs=cfgs.standardized_obs,
            gamma=cfgs.buffer_cfgs.gamma,
            lam=cfgs.buffer_cfgs.lam,
            lam_c=cfgs.buffer_cfgs.lam_c,
            adv_estimation_method=cfgs.buffer_cfgs.adv_estimation_method,
            standardized_reward=cfgs.buffer_cfgs.standardized_reward,
            standardized_cost=cfgs.buffer_cfgs.standardized_cost,
        )
        # Set up optimizer for policy and value function
        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=cfgs.actor_lr
        )
        self.reward_critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.reward_critic, learning_rate=cfgs.critic_lr
        )
        if cfgs.use_cost:
            self.cost_critic_optimizer = core.set_optimizer(
                'Adam', module=self.actor_critic.cost_critic, learning_rate=cfgs.critic_lr
            )
        # Set up scheduler for policy learning rate decay
        self.scheduler = self.set_learning_rate_scheduler()
        # Set up model saving
        what_to_save = {
            'pi': self.actor_critic.actor,
            'obs_oms': self.actor_critic.obs_oms,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # Setup statistics
        self.start_time = time.time()
        self.logger.log('Start with training.')
        self.epoch_time = None
        self.penalty_param = None
        self.p_dist = None

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

    def set_mpi(self):
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
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for CPO, etc
        """

    def check_distributed_parameters(self):
        """
        Check if parameters are synchronized across all processes.
        """

        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {
                'Policy': self.actor_critic.actor.net,
                'Value': self.actor_critic.reward_critic.net,
            }
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
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

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
        for epoch in range(self.cfgs.epochs):
            self.epoch_time = time.time()
            # Update internals of AC
            if self.cfgs.exploration_noise_anneal:
                self.actor_critic.anneal_exploration(frac=epoch / self.cfgs.epochs)
            if self.cfgs.reward_penalty:
                # Consider reward penalty parameter in reward calculation: r' = r - c
                assert hasattr(self, 'lagrangian_multiplier')
                assert hasattr(self, 'lambda_range_projection')
                self.penalty_param = self.cfgs.penalty_param
            else:
                self.penalty_param = 0.0
            # Collect data from environment
            self.env.set_rollout_cfgs(
                local_steps_per_epoch=self.local_steps_per_epoch,
                penalty_param=self.penalty_param,
                use_cost=self.cfgs.use_cost,
                cost_gamma=self.cfgs.cost_gamma,
            )
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
            )
            # Update: actor, critic, running statistics
            raw_data, _ = self.update()

            # Update running statistics, e.g. observation standardization
            # Note: observations from are raw outputs from environment
            if self.cfgs.standardized_obs:
                self.actor_critic.obs_oms.update(raw_data['obs'])
            # Apply Implement Reward scaling
            if self.cfgs.scale_rewards:
                self.actor_critic.ret_oms.update(raw_data['discounted_ret'])

            # Log and store information
            self.log(epoch)
            # Check if all models own the same parameter values
            if epoch % self.cfgs.check_freq == 0:
                self.check_distributed_parameters()
            # Save model to disk
            if (epoch + 1) % self.cfgs.save_freq == 0:
                self.logger.torch_save(itr=epoch)

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.actor_critic

    def log(self, epoch: int):
        """Log information about epoch"""
        total_env_steps = (epoch + 1) * self.cfgs.steps_per_epoch
        fps = self.cfgs.steps_per_epoch / (time.time() - self.epoch_time)
        # Step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs.linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs.actor_lr

        self.logger.log_tabular('Train/Epoch', epoch + 1)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')

        # Log information about actor
        self.logger.log_tabular('Loss/Loss_pi')
        self.logger.log_tabular('Loss/Delta_loss_pi')
        self.logger.log_tabular('Values/Adv')

        # Log information about critic
        self.logger.log_tabular('Loss/Loss_reward_critic')
        self.logger.log_tabular('Loss/Delta_loss_reward_critic')
        self.logger.log_tabular('Values/V')

        if self.cfgs.use_cost:
            # Log information about cost critic
            self.logger.log_tabular('Loss/Loss_cost_critic')
            self.logger.log_tabular('Loss/Delta_loss_cost_critic')
            self.logger.log_tabular('Values/C')

        self.logger.log_tabular('Train/Entropy')
        self.logger.log_tabular('Train/KL')
        self.logger.log_tabular('Train/StopIter')
        self.logger.log_tabular('Train/PolicyRatio')
        self.logger.log_tabular('Train/LR', current_lr)
        if self.cfgs.scale_rewards:
            reward_scale_mean = self.actor_critic.ret_oms.mean.item()
            reward_scale_stddev = self.actor_critic.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.cfgs.exploration_noise_anneal:
            noise_std = self.actor_critic.actor.std
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        if self.cfgs.model_cfgs.ac_kwargs.pi.actor_type == 'gaussian_learning':
            self.logger.log_tabular('Misc/ExplorationNoiseStd', self.actor_critic.actor.std)
        # Some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()

    def update(self):
        """
        Update actor, critic, running statistics
        """
        raw_data, data = self.buf.pre_process_data()
        # Update critic using epoch data
        self.update_value_net(data=data)
        # Update cost critic using epoch data
        if self.cfgs.use_cost:
            self.update_cost_net(data=data)
        # Update actor using epoch data
        self.update_policy_net(data=data)

        return raw_data, data

    def update_policy_net(self, data) -> None:
        """update policy network"""
        # Get prob. distribution before updates: used to measure KL distance
        with torch.no_grad():
            self.p_dist = self.actor_critic.actor(data['obs'])

        # Get loss and info values before update
        pi_l_old, _ = self.compute_loss_pi(data=data)
        loss_pi_before = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs.actor_iters):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            # Apply L2 norm
            if self.cfgs.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
                )

            # Average grads across MPI processes
            distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
            self.actor_optimizer.step()

            q_dist = self.actor_critic.actor(data['obs'])
            torch_kl = torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()

            if self.cfgs.kl_early_stopping:
                # Average KL for consistent early stopping across processes
                if distributed_utils.mpi_avg(torch_kl) > self.cfgs.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(
            **{
                'Loss/Loss_pi': loss_pi.item(),
                'Loss/Delta_loss_pi': loss_pi.item() - loss_pi_before,
                'Train/StopIter': i + 1,
                'Values/Adv': data['adv'].numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': torch_kl,
                'Train/PolicyRatio': pi_info['ratio'],
            }
        )

    def update_value_net(self, data: dict) -> None:
        """update value network"""
        # Divide whole local epoch data into mini_batches
        mbs = self.local_steps_per_epoch // self.cfgs.num_mini_batches
        assert mbs >= 16, f'Batch size {mbs}<16'

        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss_v_before = loss_fn(
            self.actor_critic.reward_critic(data['obs']), data['target_v']
        ).item()

        indices = np.arange(self.local_steps_per_epoch)
        val_losses = []
        for _ in range(self.cfgs.critic_iters):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.reward_critic_optimizer.zero_grad()
                loss_v = loss_fn(
                    self.actor_critic.reward_critic(data['obs'][mb_indices]),
                    data['target_v'][mb_indices],
                )

                loss_v.backward()
                val_losses.append(loss_v.item())
                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.actor_critic.reward_critic.net)
                self.reward_critic_optimizer.step()

        self.logger.store(
            **{
                'Loss/Delta_loss_reward_critic': np.mean(val_losses) - loss_v_before,
                'Loss/Loss_reward_critic': np.mean(val_losses),
            }
        )

    def update_cost_net(self, data: dict) -> None:
        """Some child classes require additional updates"""
        assert self.cfgs.use_cost, 'Must use cost to update cost network.'
        assert hasattr(self, 'cost_critic_optimizer')
        assert 'target_c' in data, f'provided keys: {data.keys()}'

        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss_c_before = loss_fn(self.actor_critic.cost_critic(data['obs']), data['target_c']).item()

        # Divide whole local epoch data into mini_batches
        mbs = self.local_steps_per_epoch // self.cfgs.num_mini_batches
        assert mbs >= 16, f'Batch size {mbs}<16'

        indices = np.arange(self.local_steps_per_epoch)
        losses = []

        # Train cost value network
        for _ in range(self.cfgs.critic_iters):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                # Iterate mini batch times
                end = start + mbs
                mb_indices = indices[start:end]

                self.cost_critic_optimizer.zero_grad()

                loss_c = loss_fn(
                    self.actor_critic.cost_critic(data['obs'][mb_indices]),
                    data['target_c'][mb_indices],
                )
                loss_c.backward()
                losses.append(loss_c.item())
                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.actor_critic.cost_critic.net)
                self.cost_critic_optimizer.step()

        self.logger.store(
            **{
                'Loss/Delta_loss_cost_critic': np.mean(losses) - loss_c_before,
                'Loss/Loss_cost_critic': np.mean(losses),
            }
        )
