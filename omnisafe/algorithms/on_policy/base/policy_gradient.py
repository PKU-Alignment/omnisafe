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
from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn as nn

from omnisafe.algorithms import registry
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.record_queue import RecordQueue
from omnisafe.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import core, distributed_utils
from omnisafe.utils.config_utils import dict2namedtuple, namedtuple2dict, recursive_update
from omnisafe.utils.tools import get_flat_params_from
from omnisafe.wrappers import wrapper_registry


@registry.register
# pylint: disable-next=too-many-instance-attributes
class PolicyGradient:
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `Policy Gradient <https://proceedings.neurips.cc/paper
        /1999/file/64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize PolicyGradient.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
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

        assert self.cfgs.steps_per_epoch % distributed_utils.num_procs() == 0
        self.steps_per_epoch = self.cfgs.steps_per_epoch
        self.local_steps_per_epoch = (
            cfgs.steps_per_epoch // cfgs.env_cfgs.num_envs // distributed_utils.num_procs()
        ) + 1

        # ensure local each local process can experience at least one complete episode
        assert self.env.rollout_data.max_ep_len <= self.local_steps_per_epoch, (
            f'Reduce number of cores ({distributed_utils.num_procs()})'
            f'or reduce the number of parallel envrionments {self.env.cfgs.num_envs}'
            f'or increase batch size {self.cfgs.steps_per_epoch}.'
        )

        # set up logger and save configuration to disk
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(namedtuple2dict(cfgs))
        # setup actor-critic module
        self.actor_critic = ConstraintActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            model_cfgs=cfgs.model_cfgs,
        ).to(self.device)
        self.set_mpi()
        # set up experience buffer

        self.buf = VectorOnPolicyBuffer(
            obs_space=self.env.observation_space,
            act_space=self.env.action_space,
            size=self.local_steps_per_epoch,
            gamma=cfgs.buffer_cfgs.gamma,
            lam=cfgs.buffer_cfgs.lam,
            lam_c=cfgs.buffer_cfgs.lam_c,
            advantage_estimator=cfgs.buffer_cfgs.adv_estimation_method,
            standardized_adv_r=cfgs.buffer_cfgs.standardized_rew_adv,
            standardized_adv_c=cfgs.buffer_cfgs.standardized_cost_adv,
            penalty_coefficient=cfgs.penalty_param,
            num_envs=cfgs.env_cfgs.num_envs,
            device=self.device,
        )
        # set up optimizer for policy and value function
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
        # set up scheduler for policy learning rate decay
        self.scheduler = self.set_learning_rate_scheduler()
        # set up model saving
        what_to_save = {
            'pi': self.actor_critic.actor,
            'obs_normalizer': self.env.obs_normalizer,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # set up statistics
        self.start_time = time.time()
        self.logger.log('Start with training.')
        self.epoch_time = None
        self.penalty_param = None
        self.critic_loss_fn = nn.MSELoss()
        self.loss_record = RecordQueue('loss_pi', 'loss_v', 'loss_c', maxlen=100)

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

    def set_learning_rate_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Set up learning rate scheduler.

        If use linear learning rate decay,
        the learning rate will be annealed linearly.
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

    def set_mpi(self) -> None:
        """Initialize MPI specifics.

        Sync parameters of actor and critic across cores,
        only once necessary."""
        if distributed_utils.num_procs() > 1:
            # avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            start = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # sync parameters across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-start:0.3f} sec.)')

    def algorithm_specific_logs(self) -> None:
        """Use this method to collect log information.

        e.g. log lagrangian for lagrangian-base algorithms,

        .. code-block:: python

            self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
        """

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {
                'Policy': self.actor_critic.actor,
                'Value': self.actor_critic.reward_critic,
            }
            for key, module in modules.items():
                flat_params = get_flat_params_from(module)
                global_min = distributed_utils.mpi_min(torch.sum(flat_params))
                global_max = distributed_utils.mpi_max(torch.sum(flat_params))
                assert torch.allclose(global_min, global_max), f'{key} not synced.'

    def compute_surrogate(
        self,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv (torch.Tensor): reward advantage
            cost_adv (torch.Tensor): cost advantage
        """
        return adv - 0.0 * cost_adv

    # pylint: disable-next=too-many-arguments
    def compute_loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\mathbb{E}_{s_t \sim \rho_\theta} \left[
                \sum_{t=0}^T \left( \frac{\pi_\theta ^{'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \right)
                \left( \sum_{t'=t}^T \gamma^{t'-t} r_{t'} \right)
            \right]

        where :math:`\rho_\theta` is the policy distribution, :math:`\pi_\theta` is the parameters of policy network,
        :math:`a_t` is the action at time step :math:`t`, :math:`s_t` is the observation at time step :math:`t`,
        :math:`\gamma` is the discount factor, :math:`r_{t'}` is the reward at time step :math:`t'`.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log probability`` of action stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
        """
        # policy loss
        dist, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)

        loss_pi = -(ratio * adv).mean()
        # useful extra info
        approx_kl = (0.5 * (dist.mean - act) ** 2 / dist.stddev**2).mean().item()

        # compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = {'kl': approx_kl, 'ent': ent, 'ratio': ratio.mean().item()}

        return loss_pi, pi_info

    def learn(self) -> ConstraintActorCritic:
        """This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        # main loop: collect experience in env and update/log each epoch
        for epoch in range(self.cfgs.epochs):
            self.epoch_time = time.time()
            # update internals of AC
            if self.cfgs.exploration_noise_anneal:
                self.actor_critic.anneal_exploration(frac=epoch / self.cfgs.epochs)
            # collect data from environment
            self.env.set_rollout_cfgs(
                local_steps_per_epoch=self.local_steps_per_epoch,
                use_cost=self.cfgs.use_cost,
            )
            self.env.on_policy_roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
            )
            # update: actor, critic, running statistics
            self.update()
            # log and store information
            self.log(epoch)
            # check if all models own the same parameter values
            if epoch % self.cfgs.check_freq == 0:
                self.check_distributed_parameters()
            # save model to disk
            if (epoch + 1) % self.cfgs.save_freq == 0:
                self.logger.torch_save(itr=epoch)

        # close opened files to avoid number of open files overflow
        self.logger.close()
        return self.actor_critic

    def log(self, epoch: int) -> None:
        """Log info about epoch.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Train/Epoch
                -   Current epoch.
            *   -   Metrics/EpCost
                -   Average cost of the epoch.
            *   -   Metrics/EpCost
                -   Average cost of the epoch.
            *   -   Metrics/EpRet
                -   Average return of the epoch.
            *   -   Metrics/EpLen
                -   Average length of the epoch.
            *   -   Values/V
                -   Average value in :meth:`roll_out()` (from critic network) of the epoch.
            *   -   Values/C
                -   Average cost in :meth:`roll_out()` (from critic network) of the epoch.
            *   -   Values/Adv
                -   Average advantage in :meth:`roll_out()` of the epoch.
            *   -   Loss/Loss_pi
                -   Loss of the policy network.
            *   -   Loss/Delta_loss_pi
                -   Delta loss of the policy network.
            *   -   Loss/Loss_reward_critic
                -   Loss of the value network.
            *   -   Loss/Delta_loss_reward_critic
                -   Delta loss of the value network.
            *   -   Loss/Loss_cost_critic
                -   Loss of the cost network.
            *   -   Loss/Delta_loss_cost_critic
                -   Delta loss of the cost network.
            *   -   Train/Entropy
                -   Entropy of the policy network.
            *   -   Train/KL
                -   KL divergence of the policy network.
            *   -   Train/StopIters
                -   Number of iterations of the policy network.
            *   -   Train/PolicyRatio
                -   Ratio of the policy network.
            *   -   Train/LR
                -   Learning rate of the policy network.
            *   -   Misc/Seed
                -   Seed of the experiment.
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

        Args:
            epoch (int): current epoch.
        """
        total_env_steps = (epoch + 1) * self.cfgs.steps_per_epoch
        fps = self.cfgs.steps_per_epoch / (time.time() - self.epoch_time)
        # step the actor learning rate scheduler if provided
        if self.scheduler and self.cfgs.linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.cfgs.actor_lr

        self.logger.log_tabular('Train/Epoch', epoch + 1)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')

        # log information about actor
        self.logger.log_tabular('Loss/Loss_pi')
        self.logger.log_tabular('Loss/Delta_loss_pi')
        self.logger.log_tabular('Values/Adv')

        # log information about critic
        self.logger.log_tabular('Loss/Loss_reward_critic')
        self.logger.log_tabular('Loss/Delta_loss_reward_critic')
        self.logger.log_tabular('Values/V')

        if self.cfgs.use_cost:
            # log information about cost critic
            self.logger.log_tabular('Loss/Loss_cost_critic')
            self.logger.log_tabular('Loss/Delta_loss_cost_critic')
            self.logger.log_tabular('Values/C')

        self.logger.log_tabular('Train/Entropy')
        self.logger.log_tabular('Train/KL')
        self.logger.log_tabular('Train/StopIter')
        self.logger.log_tabular('Train/PolicyRatio')
        self.logger.log_tabular('Train/LR', current_lr)
        if self.cfgs.env_cfgs.normalized_rew:
            reward_norm_mean = self.env.rew_normalizer.mean.mean().item()
            reward_norm_stddev = self.env.rew_normalizer.std.mean().item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_norm_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_norm_stddev)
        if self.cfgs.exploration_noise_anneal:
            noise_std = self.actor_critic.actor.std
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        if self.cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self.logger.log_tabular('Misc/ExplorationNoiseStd', self.actor_critic.actor.std)
        # some sub-classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()

    # pylint: disable-next=too-many-locals
    def update(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        r"""Update actor, critic, running statistics, following next steps:

        -  Get the ``raw data`` and ``processed data`` from buffer

        .. note::

            ``raw data`` is the data from environment, while ``processed data`` is the data after pre-processing.

            .. list-table::

                *   -   obs
                    -   ``observaion`` stored in buffer.
                *   -   act
                    -   ``action`` stored in buffer.
                *   -   target_v
                    -   ``target value`` stored in buffer.
                *   -   target_c
                    -   ``target cost`` stored in buffer.
                *   -   log_p
                    -   ``log probability`` stored in buffer.
                *   -   adv
                    -   ``estimated advantage`` (e.g. **GAE**) stored in buffer.
                *   -   cost_adv
                    -   ``estimated cost advantage`` (e.g. **GAE**) stored in buffer.

        -  Update value net by :meth:`update_value_net()`.
        -  Update cost net by :meth:`update_cost_net()`.
        -  Update policy net by :meth:`update_policy_net()`.

        The cost and value critic network will be updated ``critic_iters`` times (always 40),
        while the policy network will be updated ``actor_iters`` times (always 80).
        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.

        """
        # get the data from buffer
        data = self.buf.get()
        obs, act, log_p, target_v, target_c, adv, cost_adv = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        # get the loss before
        loss_pi_before, loss_v_before = self.loss_record.get_mean('loss_pi', 'loss_v')
        if self.cfgs.use_cost:
            loss_c_before = self.loss_record.get_mean('loss_c')
        self.loss_record.reset('loss_pi', 'loss_v', 'loss_c')
        # compute the old distribution of policy net.
        old_dist = self.actor_critic.actor(obs)

        # load the data into the data loader.
        dataset = torch.utils.data.TensorDataset(obs, act, target_v, target_c, log_p, adv, cost_adv)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfgs.num_mini_batches, shuffle=True
        )

        # update the value net, cost net and policy net for several times.
        for i in range(self.cfgs.actor_iters):
            for _, (obs_b, act_b, target_v_b, target_c_b, log_p_b, adv_b, cost_adv_b) in enumerate(
                loader
            ):
                # update the value net.
                self.update_value_net(obs_b, target_v_b)
                # update the cost net, if use cost.
                if self.cfgs.use_cost:
                    self.update_cost_net(obs_b, target_c_b)
                # update the policy net.
                self.update_policy_net(obs_b, act_b, log_p_b, adv_b, cost_adv_b)
            # compute the new distribution of policy net.
            new_dist = self.actor_critic.actor(obs)
            # compute the KL divergence between old and new distribution.
            torch_kl = (
                torch.distributions.kl.kl_divergence(old_dist, new_dist)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            # if the KL divergence is larger than the target KL divergence, stop the update.
            if self.cfgs.kl_early_stopping and torch_kl > self.cfgs.target_kl:
                self.logger.log(f'KL early stop at the {i+1} th step.')
                break
        # log the information.
        loss_pi, loss_v = self.loss_record.get_mean('loss_pi', 'loss_v')
        self.logger.store(
            **{
                'Loss/Loss_pi': loss_pi,
                'Loss/Delta_loss_pi': loss_pi - loss_pi_before,
                'Train/StopIter': i + 1,
                'Values/Adv': adv.mean().item(),
                'Train/KL': torch_kl,
                'Loss/Delta_loss_reward_critic': loss_v - loss_v_before,
                'Loss/Loss_reward_critic': loss_v,
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
        return data

    # pylint: disable-next=too-many-locals,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        r"""Update policy network under a double for loop.

            The pseudo code is shown below:

            .. code-block:: python

                for _ in range(self.cfgs.actor_iters):
                    for _ in range(self.cfgs.num_mini_batches):
                        # Get mini-batch data
                        # Compute loss
                        # Update network

            .. warning::
                For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
                the ``KL divergence`` between the old policy and the new policy is calculated.
                And the ``KL divergence`` is used to determine whether the update is successful.
                If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log_p`` stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
            cost_adv (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        # process the advantage function.
        processed_adv = self.compute_surrogate(adv=adv, cost_adv=cost_adv)
        # compute the loss of policy net.
        loss_pi, pi_info = self.compute_loss_pi(obs=obs, act=act, log_p=log_p, adv=processed_adv)
        # log the loss of policy net.
        self.loss_record.append(loss_pi=loss_pi.mean().item())
        # update the policy net.
        self.actor_optimizer.zero_grad()
        # backward the loss of policy net.
        loss_pi.backward()
        # clip the gradient of policy net.
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
            )
        # average the gradient of policy net.
        distributed_utils.mpi_avg_grads(self.actor_critic.actor)
        self.actor_optimizer.step()
        self.logger.store(
            **{
                'Train/Entropy': pi_info['ent'],
                'Train/PolicyRatio': pi_info['ratio'],
            }
        )

    def update_value_net(
        self,
        obs: torch.Tensor,
        target_v: torch.Tensor,
    ) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::
            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.
        The pseudo code is shown below:

        .. code-block:: python

            for _ in range(self.cfgs.actor_iters):
                for _ in range(self.cfgs.num_mini_batches):
                    # Get mini-batch data
                    # Compute loss
                    # Update network

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            target_v (torch.Tensor): ``target_v`` stored in buffer.
        """
        self.reward_critic_optimizer.zero_grad()
        # compute the loss of value net.
        loss_v = self.critic_loss_fn(
            self.actor_critic.reward_critic(obs),
            target_v,
        )
        # add the norm of critic network parameters to the loss function.
        if self.cfgs.use_critic_norm:
            for param in self.actor_critic.reward_critic.parameters():
                loss_v += param.pow(2).sum() * self.cfgs.critic_norm_coeff
        # log the loss of value net.
        self.loss_record.append(loss_v=loss_v.mean().item())
        # backward
        loss_v.backward()
        # clip the gradient
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.reward_critic.parameters(), self.cfgs.max_grad_norm
            )
        distributed_utils.mpi_avg_grads(self.actor_critic.reward_critic)
        self.reward_critic_optimizer.step()

    def update_cost_net(self, obs: torch.Tensor, target_c: torch.Tensor) -> None:
        r"""Update cost network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::
            L = \frac{1}{N} \sum_{i=1}^N (\hat{C} - C)^2

        where :math:`\hat{C}` is the predicted cost and :math:`C` is the target cost.
        The pseudo code is shown below:

        .. code-block:: python

            for _ in range(self.cfgs.actor_iters):
                for _ in range(self.cfgs.num_mini_batches):
                    # Get mini-batch data
                    # Compute loss
                    # Update network

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            target_c (torch.Tensor): ``target_c`` stored in buffer.
        """
        self.cost_critic_optimizer.zero_grad()
        # compute the loss of cost net.
        loss_c = self.critic_loss_fn(
            self.actor_critic.cost_critic(obs),
            target_c,
        )
        # add the norm of critic network parameters to the loss function.
        if self.cfgs.use_critic_norm:
            for param in self.actor_critic.cost_critic.parameters():
                loss_c += param.pow(2).sum() * self.cfgs.critic_norm_coeff
        # log the loss.
        self.loss_record.append(loss_c=loss_c.mean().item())
        # backward.
        loss_c.backward()
        # clip the gradient.
        if self.cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.cost_critic.parameters(), self.cfgs.max_grad_norm
            )
        distributed_utils.mpi_avg_grads(self.actor_critic.cost_critic)
        self.cost_critic_optimizer.step()
