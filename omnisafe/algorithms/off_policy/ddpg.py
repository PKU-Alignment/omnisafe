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
from omnisafe.common.record_queue import RecordQueue
from omnisafe.models.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import dict2namedtuple, namedtuple2dict, recursive_update
from omnisafe.wrappers import wrapper_registry


@registry.register
# pylint: disable-next=too-many-instance-attributes
class DDPG:
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize DDPG.

        Args:
            env_id (str): Environment ID.
            cfgs (NamedTuple): Configuration dictionary.
        """
        self.algo = self.__class__.__name__
        self.cfgs = deepcopy(cfgs)
        self.wrapper_type = self.cfgs.wrapper_type
        self.device = (
            f'cuda:{self.cfgs.device_id}'
            if torch.cuda.is_available() and self.cfgs.device == 'cuda'
            else 'cpu'
        )
        added_cfgs = self._get_added_cfgs()
        env_cfgs = recursive_update(
            namedtuple2dict(self.cfgs.env_cfgs), added_cfgs, add_new_args=True
        )
        env_cfgs = dict2namedtuple(env_cfgs)

        self.env = wrapper_registry.get(self.wrapper_type)(env_id, cfgs=env_cfgs)
        # set up for learning and rolling out schedule
        self.local_steps_per_epoch = (
            cfgs.steps_per_epoch // cfgs.env_cfgs.num_envs // distributed_utils.num_procs() + 1
        )
        self.total_steps = self.cfgs.epochs * self.cfgs.steps_per_epoch
        # the steps in each process should be integer
        assert cfgs.steps_per_epoch % distributed_utils.num_procs() == 0
        # ensure local each local process can experience at least one complete episode
        assert self.env.rollout_data.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()}) or increase '
            f'batch size {self.cfgs.steps_per_epoch}.'
        )
        # ensure valid number for iteration
        assert cfgs.update_every > 0
        self.max_ep_len = self.env.rollout_data.max_ep_len

        self.env.set_rollout_cfgs(
            determinstic=False,
            use_cost=self.cfgs.use_cost,
        )

        # set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))
        # set seed
        seed = int(cfgs.seed) + 10000 * distributed_utils.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # setup actor-critic module
        self.actor_critic = ConstraintActorQCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            model_cfgs=cfgs.model_cfgs,
        ).to(self.device)
        # set up experience buffer
        # obs_dim, act_dim, size, batch_size
        self.buf = BaseBuffer(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            size=cfgs.replay_buffer_cfgs.size,
            batch_size=cfgs.replay_buffer_cfgs.batch_size,
            num_envs=cfgs.env_cfgs.num_envs,
            device=self.device,
        )
        # set up optimizer for policy and q-function
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
        # set up scheduler for policy learning rate decay
        self.scheduler = self.set_learning_rate_scheduler()
        # set up target network for off_policy training
        self._ac_training_setup()
        torch.set_num_threads(10)
        # set up model saving
        what_to_save = {
            'pi': self.actor_critic.actor,
            'obs_normalizer': self.env.obs_normalizer,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # set up timer
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.logger.log('Start with training.')
        self.loss_record = RecordQueue('loss_pi', 'loss_q', 'loss_c', maxlen=100)
        self.cost_limit = None

    def _get_added_cfgs(self) -> dict:
        """Get additional configurations.

        Returns:
            dict: The additional configurations.
        """
        added_configs = {
            'device': f'cuda:{self.cfgs.device_id}'
            if torch.cuda.is_available() and self.cfgs.device == 'cuda'
            else 'cpu',
            'seed': self.cfgs.seed,
        }
        return added_configs

    def cost_limit_decay(
        self,
        epoch: int,
        end_epoch: int,
    ) -> None:
        """Decay cost limit."""
        if epoch < end_epoch:
            assert hasattr(self, 'cost_limit'), 'Cost limit is not set.'
            self.cost_limit = (
                self.cfgs.init_cost_limit * (1 - epoch / end_epoch)
                + self.cfgs.target_cost_limit * epoch / end_epoch
            )

    def set_learning_rate_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Set up learning rate scheduler.

        If use linear learning rate decay, the learning rate will be annealed linearly.
        """
        scheduler = None
        if self.cfgs.linear_lr_decay:
            # linear anneal
            def linear_anneal(epoch):
                return 1 - epoch / self.cfgs.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.actor_optimizer, lr_lambda=linear_anneal
            )
        return scheduler

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
        # freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self.ac_targ.actor.parameters():
            param.requires_grad = False
        for param in self.ac_targ.critic.parameters():
            param.requires_grad = False
        for param in self.ac_targ.cost_critic.parameters():
            param.requires_grad = False

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
        action, _ = self.actor_critic.actor.predict(obs, deterministic=True)
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
        self.logger.store(
            **{
                'Train/RewardQValues': q_value.mean().item(),
            }
        )
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, _ = self.ac_targ.actor.predict(obs, deterministic=True, need_log_prob=False)
            q_targ = self.ac_targ.critic(next_obs, act_targ)[0]
            backup = rew + self.cfgs.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q_value - backup) ** 2).mean()
        # useful info for logging
        q_info = {'QVals': q_value.detach().mean().item()}
        return loss_q, q_info

    # pylint: disable-next=too-many-arguments
    def compute_loss_c(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
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
        self.logger.store(
            **{
                'Train/CostQValues': cost_q_value.mean().item(),
            }
        )
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, _ = self.ac_targ.actor.predict(obs, deterministic=False, need_log_prob=False)
            qc_targ = self.ac_targ.cost_critic(next_obs, act_targ)[0]
            backup = cost + self.cfgs.gamma * qc_targ
        # MSE loss against Bellman backup
        loss_qc = ((cost_q_value - backup) ** 2).mean()
        # useful info for logging
        qc_info = {'QCosts': cost_q_value.detach().mean().item()}

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
        for steps in range(
            0, self.local_steps_per_epoch * self.cfgs.epochs, self.cfgs.update_every
        ):
            # until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            use_rand_action = steps < self.cfgs.start_steps
            roll_out_steps = steps % self.cfgs.steps_per_epoch
            self.env.off_policy_roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=False,
                use_rand_action=use_rand_action,
                ep_steps=self.cfgs.update_every,
            )

            # update handling
            if steps >= self.cfgs.update_after:
                for _ in range(self.cfgs.update_every):
                    batch = self.buf.sample_batch()
                    self.update(data=batch)

            # end of epoch handling
            if (roll_out_steps + self.cfgs.update_every) >= self.cfgs.steps_per_epoch:
                epoch = steps // self.cfgs.steps_per_epoch + 1
                if self.cfgs.cost_limit_decay:
                    self.cost_limit_decay(epoch, self.cfgs.end_epoch)
                if self.cfgs.exploration_noise_anneal:
                    self.actor_critic.anneal_exploration(frac=epoch / self.cfgs.epochs)

                # save model to disk
                if (epoch + 1) % self.cfgs.save_freq == 0:
                    self.logger.torch_save(itr=epoch)
                # log info about epoch
                self.test_agent()
                self.log(epoch, steps)
        return self.actor_critic

    def update(self, data: dict) -> None:
        r"""Update actor, critic, running statistics, following next steps:

        -  Get the ``data`` from buffer

        .. note::

            .. list-table::

                *   -   obs
                    -   ``observaion`` stored in buffer.
                *   -   act
                    -   ``action`` stored in buffer.
                *   -   rew
                    -   ``reward`` stored in buffer.
                *   -   cost
                    -   ``cost`` stored in buffer.
                *   -   next_obs
                    -   ``next observaion`` stored in buffer.
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
        # get the loss before
        loss_pi_before, loss_q_before = self.loss_record.get_mean('loss_pi', 'loss_q')
        if self.cfgs.use_cost:
            loss_c_before = self.loss_record.get_mean('loss_c')
        self.loss_record.reset('loss_pi', 'loss_q', 'loss_c')
        # first run one gradient descent step for Q.
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
            )
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = False

        # freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

        # next run one gradient descent step for actor.
        self.update_policy_net(obs=obs)

        # unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # finally, update target networks by polyak averaging.
        self.polyak_update_target()
        loss_pi, loss_q = self.loss_record.get_mean('loss_pi', 'loss_q')
        self.logger.store(
            **{
                'Loss/Loss_pi': loss_pi,
                'Loss/Delta_loss_pi': loss_pi - loss_pi_before,
                'Loss/Delta_loss_reward_critic': loss_q - loss_q_before,
                'Loss/Loss_reward_critic': loss_q,
            }
        )
        if self.cfgs.use_cost:
            loss_c = self.loss_record.get_mean('loss_c')
            self.logger.store(
                **{
                    'Loss/Delta_loss_cost_critic': loss_c - loss_c_before,
                    'Loss/Loss_cost_critic': loss_c,
                }
            )

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
                # notes: We use an in-place operations "mul_", "add_" to update target
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
        # train policy with one steps of gradient descent
        self.actor_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(obs)
        # log the loss of policy net.
        self.loss_record.append(loss_pi=loss_pi.mean().item())
        loss_pi.backward()
        # clip the gradient of policy net.
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
            )
        self.actor_optimizer.step()

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
        # train value critic with one steps of gradient descent
        self.critic_optimizer.zero_grad()
        loss_q, _ = self.compute_loss_v(
            obs=obs,
            act=act,
            rew=rew,
            next_obs=next_obs,
            done=done,
        )
        # add the norm of critic network parameters to the loss function.
        if self.cfgs.use_critic_norm:
            for param in self.actor_critic.critic.parameters():
                loss_q += param.pow(2).sum() * self.cfgs.critic_norm_coeff
        # log the loss of value net.
        self.loss_record.append(loss_q=loss_q.mean().item())
        loss_q.backward()
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.critic.parameters(), self.cfgs.max_grad_norm
            )
        self.critic_optimizer.step()

    # pylint: disable-next=too-many-arguments
    def update_cost_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        r"""Update cost network.

        - Get the loss of policy network.
        - Update policy network by loss.
        - Log useful information.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            cost (:class:`torch.Tensor`): ``cost`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observation`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        # train cost critic with one steps of gradient descent
        self.cost_critic_optimizer.zero_grad()
        loss_qc, _ = self.compute_loss_c(
            obs=obs,
            act=act,
            cost=cost,
            next_obs=next_obs,
        )
        # add the norm of critic network parameters to the loss function.
        if self.cfgs.use_critic_norm:
            for param in self.actor_critic.cost_critic.parameters():
                loss_qc += param.pow(2).sum() * self.cfgs.critic_norm_coeff
        # log the loss of value net.
        self.loss_record.append(loss_c=loss_qc.mean().item())
        loss_qc.backward()
        # clip the gradient.
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.cost_critic.parameters(), self.cfgs.max_grad_norm
            )
        self.cost_critic_optimizer.step()

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
        # step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs.linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs.actor_lr

        self.logger.log_tabular('Train/Epoch', epoch)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')

        self.logger.log_tabular('Test/EpRet')
        self.logger.log_tabular('Test/EpCost')
        self.logger.log_tabular('Test/EpLen')
        # log information about actor
        self.logger.log_tabular('Loss/Loss_pi')
        self.logger.log_tabular('Loss/Delta_loss_pi')

        # log information about critic
        self.logger.log_tabular('Loss/Loss_reward_critic')
        self.logger.log_tabular('Loss/Delta_loss_reward_critic')
        self.logger.log_tabular('Values/V')
        self.logger.log_tabular('Train/RewardQValues')

        if self.cfgs.use_cost:
            # log information about cost critic
            self.logger.log_tabular('Loss/Loss_cost_critic')
            self.logger.log_tabular('Loss/Delta_loss_cost_critic')
            self.logger.log_tabular('Values/C')
            self.logger.log_tabular('Train/CostQValues')

        self.logger.log_tabular('Misc/Seed', self.cfgs.seed)
        self.logger.log_tabular('LR', current_lr)
        if self.cfgs.env_cfgs.normalized_rew:
            reward_norm_mean = self.env.rew_normalizer.mean.mean().item()
            reward_norm_stddev = self.env.rew_normalizer.std.mean().item()
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

    def test_agent(self):
        """Test agent."""
        for _ in range(self.cfgs.num_test_episodes):
            self.env.off_policy_roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=True,
                use_rand_action=False,
                ep_steps=self.max_ep_len,
                is_train=False,
            )
