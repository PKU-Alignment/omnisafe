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
from typing import Dict, NamedTuple, Tuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.common.base_buffer import VectorBaseBuffer as BaseBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.utils.tools import get_flat_params_from
from omnisafe.wrappers import wrapper_registry


@registry.register
class DDPG:  # pylint: disable=too-many-instance-attributes
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez,
                   Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize DDPG.

        Args:
            env_id (str): Environment ID.
            cfgs (NamedTuple): Configuration dictionary.
        """
        self.cfgs = deepcopy(cfgs)
        self.wrapper_type = self.cfgs.wrapper_type
        self.env = wrapper_registry.get(self.wrapper_type)(
            env_id, cfgs=self.cfgs._asdict().get('env_cfgs')
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
        assert self.env.rollout_data.max_ep_len <= self.local_steps_per_epoch, (
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
            use_cost=self.cfgs.use_cost,
        )

        # Set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))
        # Set seed
        seed = cfgs.seed + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Setup actor-critic module
        self.actor_critic = ConstraintActorQCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
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
            num_envs=cfgs.env_cfgs.num_envs,
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
            'obs_normalize': self.env.obs_normalize,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # Set up timer
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.logger.log('Start with training.')

    def set_learning_rate_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Set up learning rate scheduler.

        If use linear learning rate decay, the learning rate will be annealed linearly.
        """
        scheduler = None
        if self.cfgs.linear_lr_decay:
            # Linear anneal
            def linear_anneal(epoch):
                return 1 - epoch / self.cfgs.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.actor_optimizer, lr_lambda=linear_anneal
            )
        return scheduler

    def _init_mpi(self) -> None:
        """Initialize MPI specifics."""
        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            start = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync params across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def algorithm_specific_logs(self) -> None:
        """Use this method to collect log information.

        e.g. log lagrangian for lagrangian-base algorithms, etc.
        """

    def _ac_training_setup(self) -> None:
        """Set up target network for off_policy training.

        Set target network,
        which is frozen and can only be updated via polyak averaging,
        by calling :meth`polyak_update`.
        """
        self.ac_targ = deepcopy(self.actor_critic)
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self.ac_targ.actor.parameters():
            param.requires_grad = False
        for param in self.ac_targ.critic.parameters():
            param.requires_grad = False
        for param in self.ac_targ.cost_critic.parameters():
            param.requires_grad = False

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {'Policy': self.actor_critic.actor.net, 'Value': self.actor_critic.critic.net}
            for key, module in modules.items():
                flat_params = get_flat_params_from(module).numpy()
                global_min = distributed_utils.mpi_min(np.sum(flat_params))
                global_max = distributed_utils.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing ``pi/actor`` loss.

        Detailedly, the loss function in DDPG is defined as:

        .. math::
            L = -Q^V(s, \pi(s))

        where :math:`Q^V` is the reward critic network,
        and :math:`\pi` is the policy network.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
        """
        action = self.actor_critic.actor.predict(obs, deterministic=False)
        loss_pi = self.actor_critic.critic(obs, action)[0]
        pi_info = {}
        return -loss_pi.mean(), pi_info

    # pylint: disable-next=too-many-arguments
    def compute_loss_v(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing value loss.

        Specifically, the loss function in DDPG is defined as:

        .. math::
            L = [Q^V(s, a) - (r + \gamma (1-d) Q^V(s', \pi(s')))]^2

        where :math:`\gamma` is the discount factor, :math:`d` is the terminal signal,
        :math:`Q^V` is the value critic function, and :math:`\pi` is the policy.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            rew (:class:`torch.Tensor`): ``reward`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observation`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        q_value = self.actor_critic.critic(obs, act)[0]
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ = self.ac_targ.actor.predict(obs, deterministic=True, need_log_prob=False)
            q_targ = self.ac_targ.critic(next_obs, act_targ)[0]
            backup = rew + self.cfgs.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q_value - backup) ** 2).mean()
        # Useful info for logging
        q_info = dict(QVals=q_value.detach().numpy())
        return loss_q, q_info

    # pylint: disable-next=too-many-arguments
    def compute_loss_c(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing cost critic loss.

        Specifically, the loss function is defined as:

        .. math::
            L = [Q^C(s, a) - (c + \gamma (1-d) Q^C(s', \pi(s')))]^2

        where :math:`\gamma` is the discount factor, :math:`d` is the termination signal,
        :math:`Q^C` is the cost critic function, and :math:`\pi` is the policy.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            cost (:class:`torch.Tensor`): ``cost`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observation`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        cost_q_value = self.actor_critic.cost_critic(obs, act)[0]

        # Bellman backup for Q function
        with torch.no_grad():
            action = self.ac_targ.actor.predict(next_obs, deterministic=True)
            qc_targ = self.ac_targ.cost_critic(next_obs, action)[0]
            backup = cost + self.cfgs.gamma * (1 - done) * qc_targ
        # MSE loss against Bellman backup
        loss_qc = ((cost_q_value - backup) ** 2).mean()
        # Useful info for logging
        qc_info = dict(QCosts=cost_q_value.detach().numpy())

        return loss_qc, qc_info

    def learn(self) -> ConstraintActorQCritic:
        r"""This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.

        .. note::
            If you want to customize the algorithm, you can rewrite this function.
            For example, In the Lagrange version of DDPG algorithm,
            we need to update the Lagrange multiplier.
            So a new function :meth:`update_lagrange_multiplier` is added.
            For details, sees in :class:`DDPGLag`.

        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        for steps in range(0, self.local_steps_per_epoch * self.epochs, self.update_every):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            use_rand_action = steps < self.start_steps
            self.env.off_policy_roll_out(
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
                # Log info about epoch
                self.log(epoch, steps)
        return self.actor_critic

    def update(self, data: dict) -> None:
        r"""Update actor, critic, running statistics, following next steps:

        -  Get the ``data`` from buffer

        .. note::

            .. list-table::

                *   -   obs
                    -   ``obsertaion`` stored in buffer.
                *   -   act
                    -   ``action`` stored in buffer.
                *   -   rew
                    -   ``reward`` stored in buffer.
                *   -   cost
                    -   ``cost`` stored in buffer.
                *   -   next_obs
                    -   ``next obsertaion`` stored in buffer.
                *   -   done
                    -   ``terminated`` stored in buffer.

        -  Update value net by :meth:`update_value_net`.
        -  Update cost net by :meth:`update_cost_net`.
        -  Update policy net by :meth:`update_policy_net`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        """
        # First run one gradient descent step for Q.
        obs, act, rew, cost, next_obs, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['cost'],
            data['next_obs'],
            data['done'],
        )
        self.update_value_net(
            obs=obs,
            act=act,
            rew=rew,
            next_obs=next_obs,
            done=done,
        )
        if self.cfgs.use_cost:
            self.update_cost_net(
                obs=obs,
                act=act,
                cost=cost,
                next_obs=next_obs,
                done=done,
            )
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = False

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for actor.
        self.update_policy_net(obs=obs)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()

    def polyak_update_target(self) -> None:
        r"""One important trick is to use a target network to stabilize the training of the Q-function.

        .. note::
            This is done by updating the target network according to:

            .. math::
                \theta_{\text{targ}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{targ}}

            where :math:`\tau` is a small number (e.g. 0.005),
            and :math:`\theta` are the parameters of the Q-network.
            This is called a `polyak averaging <https://en.wikipedia.org/wiki/>`_
        """
        with torch.no_grad():
            for param, param_targ in zip(self.actor_critic.parameters(), self.ac_targ.parameters()):
                # Notes: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                param_targ.data.mul_(self.cfgs.polyak)
                param_targ.data.add_((1 - self.cfgs.polyak) * param.data)

    def update_policy_net(self, obs: torch.Tensor) -> None:
        """Update policy network.

        - Get the loss of policy network.
        - Update policy network by loss.
        - Log useful information.

        Args:
            obs (:class:`torch.Tensor`): observation.
        """
        # Train policy with one steps of gradient descent
        self.actor_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(obs)
        loss_pi.backward()
        self.actor_optimizer.step()
        self.logger.store(**{'Loss/Pi': loss_pi.item()})

    # pylint: disable-next=too-many-arguments
    def update_value_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Update value network.

        - Get the loss of policy network.
        - Update policy network by loss.
        - Log useful information.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            rew (:class:`torch.Tensor`): ``reward`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observation`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        # Train value critic with one steps of gradient descent
        self.critic_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_v(
            obs=obs,
            act=act,
            rew=rew,
            next_obs=next_obs,
            done=done,
        )
        loss_q.backward()
        self.critic_optimizer.step()
        self.logger.store(**{'Loss/Value': loss_q.item(), 'QVals': q_info['QVals']})

    # pylint: disable-next=too-many-arguments
    def update_cost_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        r"""Update cost network.

        - Get the loss of policy network.
        - Update policy network by loss.
        - Log useful information.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            cost (:class:`torch.Tensor`): ``cost`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observations`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        # Train cost critic with one steps of gradient descent
        self.cost_critic_optimizer.zero_grad()
        loss_qc, qc_info = self.compute_loss_c(
            obs=obs,
            act=act,
            cost=cost,
            next_obs=next_obs,
            done=done,
        )
        loss_qc.backward()
        self.cost_critic_optimizer.step()
        self.logger.store(**{'Loss/Cost': loss_qc.item(), 'QCosts': qc_info['QCosts']})

    def test_agent(self) -> None:
        """Test agent.

        The algorithm uses a randomness strategy during training,
        and a deterministic strategy during testing.
        """
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

    def log(self, epoch: int, total_steps: int) -> None:
        """Log info about epoch.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Epoch
                -   Current epoch.
            *   -   Metrics/EpCost
                -   Average cost of the epoch.
            *   -   Metrics/EpCost
                -   Average cost of the epoch.
            *   -   Metrics/EpRet
                -   Average return of the epoch.
            *   -   Metrics/EpLen
                -   Average length of the epoch.
            *   -   Test/EpRet
                -   Average return of the test.
            *   -   Test/EpCost
                -   Average cost of the test.
            *   -   Test/EpLen
                -   Average length of the test.
            *   -   Values/V
                -   Average value in :meth:`roll_out` (from critic network) of the epoch.
            *   -   Values/Cost
                -   Average cost in :meth:`roll_out` (from critic network) of the epoch.
            *   -   Values/QVals
                -   Average Q value in :meth:`roll_out` (from critic network) of the epoch.
            *   -   Values/QCosts
                -   Average Q cost in :meth:`roll_out` (from critic network) of the epoch.
            *   -   loss/Pi
                -   Loss of the policy network.
            *   -   loss/Value
                -   Loss of the value network.
            *   -   loss/Cost
                -   Loss of the cost network.
            *   -   Misc/Seed
                -   Seed of the experiment.
            *   -   LR
                -   Learning rate of the actor network.
            *   -   Misc/RewScaleMean
                -   Mean of the reward scale.
            *   -   Misc/RewScaleStddev
                -   Std of the reward scale.
            *   -   Misc/ExplorationNoisestd
                -   Std of the exploration noise.
            *   -   Misc/TotalEnvSteps
                -   Total steps of the experiment.
            *   -   Time
                -   Total time.
            *   -   FPS
                -   Frames per second of the epoch.
        """
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
        if self.cfgs.env_cfgs.normalized_rew:
            reward_norm_mean = self.env.rew_normalize.mean.mean().item()
            reward_norm_stddev = self.env.rew_normalize.std.mean().item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_norm_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_norm_stddev)
        if self.cfgs.exploration_noise_anneal:
            noise_std = np.exp(self.actor_critic.actor.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()
