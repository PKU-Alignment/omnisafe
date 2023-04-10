# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Deep Deterministic Policy Gradient algorithm."""

import time
from typing import Any, Dict, Tuple, Union, Optional

from rich.progress import track

import torch
from torch import nn

from omnisafe.adapter import ModelBasedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger

from omnisafe.algorithms.model_based.models import EnsembleDynamicsModel
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.common.buffer import OnPolicyBuffer

from omnisafe.utils import distributed
from omnisafe.algorithms.model_based.base import PETS
import numpy as np
from matplotlib import pylab
from gymnasium.utils.save_video import save_video
import os


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class MBPPO(PETS):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def _init_env(self) -> None:
        self._env = ModelBasedAdapter(
            self._env_id, 1, self._seed, self._cfgs
        )
        self._env_auxiliary = ModelBasedAdapter(
            self._env_id, 1, self._seed, self._cfgs
        )

        assert int(self._cfgs.train_cfgs.total_steps) % self._cfgs.logger_cfgs.log_cycle == 0
        self._total_steps = int(self._cfgs.train_cfgs.total_steps)
        self._steps_per_epoch = int(self._cfgs.logger_cfgs.log_cycle)
        self._epochs = self._total_steps // self._cfgs.logger_cfgs.log_cycle

    def _init_model(self) -> None:
        self._dynamics_state_space = self._env.coordinate_observation_space if self._env.coordinate_observation_space is not None else self._env.observation_space
        self._dynamics = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_size=self._dynamics_state_space.shape[0],
            action_size=self._env.action_space.shape[0],
            reward_size=1,
            cost_size=1,
            use_cost=False,
            use_terminal=False,
            use_var=False,
            use_reward_critic=False,
            use_cost_critic=False,
            actor_critic=None,
            rew_func=None,
            cost_func=None,
            terminal_func=None,
        )
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

        self._use_actor_critic = True
        #self._policy_state_space = self._env.lidar_observation_space if self._env.lidar_observation_space is not None else self._env.observation_space
        self._policy_state_space = self._env.coordinate_observation_space if self._env.coordinate_observation_space is not None else self._env.observation_space

        self._actor_critic = ConstraintActorCritic(
            obs_space=self._policy_state_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        # Set up optimizer for policy and value function


    def _init(self) -> None:
        #self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self._dynamics_buf = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.dynamics_cfgs.batch_size,
            device=self._device,
        )


        self._policy_buf = OnPolicyBuffer(
            obs_space=self._policy_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.imaging_steps_per_policy_update,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            device=self._device,
        )


        if self._cfgs.evaluation_cfgs.use_eval:
            self._eval_fn = self._evaluation_single_step
        else:
            self._eval_fn = None

    def _init_log(self) -> None:
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: Dict[str, Any] = {}
        # Set up model saving
        what_to_save = {
            'actor_critic': self._actor_critic,
            'dynamics': self._dynamics,
        }
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()
        self._logger.register_key('Train/Epoch')
        self._logger.register_key('TotalEnvSteps')
        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)
        if self._cfgs.evaluation_cfgs.use_eval:
            self._logger.register_key('EvalMetrics/EpRet', window_length=5)
            self._logger.register_key('EvalMetrics/EpCost', window_length=5)
            self._logger.register_key('EvalMetrics/EpLen', window_length=5)
        self._logger.register_key('Loss/DynamicsTrainMseLoss')
        self._logger.register_key('Loss/DynamicsValMseLoss')

        #self._logger.register_key('Metrics/LagrangeMultiplier')
        self.logger.register_key('DynaMetrics/EpRet')
        self.logger.register_key('DynaMetrics/EpLen')
        self.logger.register_key('DynaMetrics/EpCost')
        self.logger.register_key('Loss/DynamicsTrainMseLoss')
        self.logger.register_key('Loss/DynamicsValMseLoss')

        self.logger.register_key('Megaiter')

        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio')
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')
        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')


        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/UpdateDynamics')
        if self._use_actor_critic:
            self._logger.register_key('Time/UpdateActorCritic')
        if self._cfgs.evaluation_cfgs.use_eval:
            self._logger.register_key('Time/Eval')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def _update_policy(self, time_step):  # pylint: disable=unused-argument
        """update actor critic"""
        megaiter = 0
        last_valid_rets = np.zeros(self._cfgs.dynamics_cfgs.elite_size)
        while True:
            self.imagine_rollout(megaiter)
            # validation
            if megaiter > 0:
                old_actor = self.get_param_values(self._actor_critic.actor)
                old_reward_critic = self.get_param_values(self._actor_critic.reward_critic)
                old_cost_critic = self.get_param_values(self._actor_critic.cost_critic)
                # data = self._policy_buf.get()
                # ep_costs = self.logger.get_stats('DynaMetrics/EpCost')[0]
                # self.update_lagrange_multiplier(ep_costs)
                # self.update_policy_net(data=data)
                # self.update_value_net(data=data)
                self._update()
                result, valid_rets = self.validation(last_valid_rets)
                if result is True:
                    # backtrack
                    self.set_param_values(old_actor, self._actor_critic.actor)
                    self.set_param_values(old_reward_critic, self._actor_critic.reward_critic)
                    self.set_param_values(old_cost_critic, self._actor_critic.cost_critic)
                    megaiter += 1
                    break
                megaiter += 1
                last_valid_rets = valid_rets
            else:
                megaiter += 1
                # data = self._policy_buf.get()
                # ep_costs = self.logger.get_stats('DynaMetrics/EpCost')[0]
                # self.update_lagrange_multiplier(ep_costs)
                # self.update_policy_net(data=data)
                # self.update_value_net(data=data)
                self._update()
        self.logger.store(Megaiter=megaiter)


    def _update(self) -> None:
        r"""Update actor, critic, following next steps:

        -  Get the ``data`` from buffer

        .. hint::

            .. list-table::

                *   -   obs
                    -   ``observaion`` stored in buffer.
                *   -   act
                    -   ``action`` stored in buffer.
                *   -   target_value_r
                    -   ``target value`` stored in buffer.
                *   -   target_value_c
                    -   ``target cost`` stored in buffer.
                *   -   logp
                    -   ``log probability`` stored in buffer.
                *   -   adv
                    -   ``estimated advantage`` (e.g. **GAE**) stored in buffer.
                *   -   cost_adv
                    -   ``estimated cost advantage`` (e.g. **GAE**) stored in buffer.

        -  Update value net by :meth:`_update_reward_critic()`.
        -  Update cost net by :meth:`_update_cost_critic()`.
        -  Update policy net by :meth:`_update_actor()`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.

        Args:
            self (object): object of the class.
        """
        # # note that logger already uses MPI statistics across all processes..
        # Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # # first update Lagrange multiplier parameter
        # self._lagrange.update_lagrange_multiplier(Jc)

        data = self._policy_buf.get()
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
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            self._update_reward_critic(obs, target_value_r)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, target_value_c)
            self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Train/StopIter': i + 1,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': kl,
            },
        )


    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::
            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            target_value_r (torch.Tensor): ``target_value_r`` stored in buffer.
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store(**{'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::
            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            target_value_c (torch.Tensor): ``target_value_c`` stored in buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(**{'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        r"""Update policy network under a double for loop.

            #. Compute the loss function.
            #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
            #. Update the network by loss function.

            .. warning::

                For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
                the ``KL divergence`` between the old policy and the new policy is calculated.
                And the ``KL divergence`` is used to determine whether the update is successful.
                If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log_p`` stored in buffer.
            adv_r (torch.Tensor): ``advantage`` stored in buffer.
            adv_c (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss, info = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): reward advantage
            adv_c (torch.Tensor): cost advantage
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\mathbb{E}_{s_t \sim \rho_\theta} [
                \sum_{t=0}^T ( \frac{\pi^{'}_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_\theta` is the policy network, :math:`\pi^{'}_\theta`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            logp (torch.Tensor): ``log probability`` of action stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        info = {'entropy': entropy, 'ratio': ratio.mean().item(), 'std': std}
        return loss, info

    def get_param_values(self, model):
        """get the dynamics parameters"""
        trainable_params = list(model.parameters())
        params = np.concatenate(
            [p.contiguous().view(-1).data.cpu().numpy() for p in trainable_params]
        )
        return params.copy()

    def set_param_values(self, new_params, model, set_new=True):
        """set the dynamics parameters"""
        trainable_params = list(model.parameters())
        param_shapes = [p.data.cpu().numpy().shape for p in trainable_params]
        param_sizes = [p.data.cpu().numpy().size for p in trainable_params]
        if set_new:
            current_idx = 0
            for idx, param in enumerate(trainable_params):
                vals = new_params[current_idx : current_idx + param_sizes[idx]]
                vals = vals.reshape(param_shapes[idx])
                param.data = torch.from_numpy(vals).float().to(self._device)
                current_idx += param_sizes[idx]

    def virtual_step(self, state, action, idx=None):
        """use virtual environment to predict next state, reward, cost"""
        if hasattr(self._env, 'task') and self._env.task == 'Goal':
            traj = self._dynamics.imagine(states=state, horizon=1, idx=idx, actions=action.unsqueeze(0))
            next_state = traj['states'][0][0]
            reward = traj['rewards'][0][0][0]
            goal_flag = self._env_auxiliary.get_goal_flag_from_obs_tensor(next_state)
            goal_flag = goal_flag.squeeze(0).cpu().item()
            #cost = traj['costs'][0][0]
            cost = torch.zeros_like(reward, device=self._device)
            info = {'goal_flag': goal_flag}
        else:
            NotImplementedError

        return next_state, reward, cost, info

    def imagine_rollout(self, megaiter):  # pylint: disable=too-many-locals
        """collect data and store to experience buffer."""
        state, _ = self._env_auxiliary.reset()
        dyna_ret, dyna_cost, dyna_len = torch.zeros(1), torch.zeros(1), torch.zeros(1)

        mix_real = self._cfgs.algo_cfgs.mixed_real_time_steps if megaiter == 0 else 0

        for time_step in range(self._cfgs.algo_cfgs.imaging_steps_per_policy_update - mix_real):
            action, action_info = self._select_action(time_step, state, self._env_auxiliary)
            next_state, reward, cost, info = self.virtual_step(state, action)

            dyna_ret += reward.cpu()
            dyna_cost += (self._cfgs.algo_cfgs.cost_gamma**dyna_len) * cost.cpu()
            dyna_len += 1

            # obs, reward, cost = expand_dims(
            #     action_info['state_vec'], reward, cost
            # )

            self._policy_buf.store(
                obs=state,
                act=action,
                reward=reward,
                cost=cost,
                value_r=action_info['val'],
                value_c=action_info['cval'],
                logp=action_info['logp'],
            )
            state = next_state

            timeout = dyna_len.item() == self._cfgs.algo_cfgs.train_horizon
            truncated = timeout
            epoch_ended = time_step == self._cfgs.algo_cfgs.imaging_steps_per_policy_update - 1
            if truncated or epoch_ended or ('goal_flag' in info.keys() and info['goal_flag']):
                if timeout or epoch_ended or ('goal_flag' in info.keys() and info['goal_flag']):
                    # state_tensor = torch.as_tensor(
                    #     action_info['state_vec'], device=self.device, dtype=torch.float32
                    # )
                    _, terminal_value, terminal_cost_value, _ = self._actor_critic.step(state)
                    # terminal_value, terminal_cost_value = torch.unsqueeze(
                    #     terminal_value, 0
                    # ), torch.unsqueeze(terminal_cost_value, 0)
                else:
                    # this means episode is terminated,
                    # and this will be triggered only in robots fall down case
                    terminal_value, terminal_cost_value = torch.zeros(
                    1, dtype=torch.float32, device=self._device
                ), torch.zeros(1, dtype=torch.float32, device=self._device)

                self._policy_buf.finish_path(terminal_value, terminal_cost_value)

                if timeout:
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(
                        **{
                            'DynaMetrics/EpRet': dyna_ret.item(),
                            'DynaMetrics/EpCost': dyna_cost.item(),
                            'DynaMetrics/EpLen': dyna_len.item(),
                        }
                    )
                state, _ = self._env_auxiliary.reset()
                dyna_ret, dyna_cost, dyna_len = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    def validation(self, last_valid_rets):
        """policy validation"""
        valid_rets = np.zeros(self._cfgs.dynamics_cfgs.elite_size)
        winner = 0
        for valid_id in range(len(valid_rets)):  # pylint:disable=consider-using-enumerate
            state, _ = self._env_auxiliary.reset()
            for step in range(self._cfgs.algo_cfgs.validation_horizon):
                action, _ = self._select_action(step, state, self._env_auxiliary)
                next_state, reward, _, info = self.virtual_step(state, action, idx=valid_id)
                valid_rets[valid_id] += reward.cpu().item()
                state = next_state
                if 'goal_flag' in info.keys() and info['goal_flag']:
                    state, _  = self._env_auxiliary.reset()
            if valid_rets[valid_id] > last_valid_rets[valid_id]:
                winner += 1
        performance_ratio = winner / self._cfgs.dynamics_cfgs.elite_size
        threshold = self._cfgs.algo_cfgs.validation_threshold_num / self._cfgs.dynamics_cfgs.elite_size
        result = performance_ratio < threshold
        return result, valid_rets

    def _select_action(
            self,
            current_step: int,
            state: torch.Tensor,
            env: ModelBasedAdapter,
            ) -> Tuple[np.ndarray, Dict]:
        """action selection"""

        # if hasattr(self._env, 'task') and self._env.task == 'Goal':
        #     state = env.get_lidar_from_coordinate(state)
        action, value_r, value_c, logp =  self._actor_critic.step(state)

        action_info = {'actor_state': state, 'val': value_r, 'cval': value_c, 'logp': logp}
        assert action.shape == torch.Size([state.shape[0], self._env.action_space.shape[0]]), "action shape should be [batch_size, action_dim]"
        info = {}
        return action, action_info

    # pylint: disable-next=too-many-arguments
    def _store_real_data(
        self,
        current_step: int,
        ep_len: int,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_state: torch.Tensor,
        info: dict,
        action_info: dict,
    ):

        """store real data"""
        done = terminated or truncated
        if 'goal_met' not in info.keys():
            goal_met = False
        else:
            goal_met = info['goal_met']
        if not terminated and not truncated and not goal_met:
            # if goal_met == true, Current goal position is not related to the last goal position, this huge transition will confuse the dynamics model.
            self._dynamics_buf.store(
                obs=state, act=action, reward=reward, cost=cost, next_obs=next_state, done=done

            )
        if (
            current_step % self._cfgs.algo_cfgs.update_policy_cycle <= self._cfgs.algo_cfgs.mixed_real_time_steps
            and self._policy_buf.ptr < self._cfgs.algo_cfgs.mixed_real_time_steps
        ):
            # obs, reward, cost = expand_dims(
            #     action_info['state_vec'], reward, cost
            # )
            #print(torch.tensor(obs,device=self._cfgs.algo_cfgs.device),raw_action,torch.tensor(reward,device=self._cfgs.algo_cfgs.device),action_info['val'].unsqueeze(0),action_info['logp'],torch.tensor(cost,device=self._cfgs.algo_cfgs.device),action_info['cval'].unsqueeze(0))
            self._policy_buf.store(
                obs=state,
                act=action,
                reward=reward,
                value_r=action_info['val'],
                logp=action_info['logp'],
                cost=cost,
                value_c=action_info['cval'],
            )
            if terminated:
                # this means episode is terminated,
                # which will be triggered only in robots fall down case
                terminal_value, terminal_cost_value = torch.zeros(
                1, dtype=torch.float32, device=self._device
            ), torch.zeros(1, dtype=torch.float32, device=self._device)

                self._policy_buf.finish_path(terminal_value, terminal_cost_value)

            # reached max imaging horizon, mixed real timestep, real max timestep , or episode truncated.
            elif (
                current_step % self._cfgs.algo_cfgs.train_horizon < self._cfgs.algo_cfgs.action_repeat
                or self._policy_buf.ptr == self._cfgs.algo_cfgs.mixed_real_time_steps
                or current_step >= self._cfgs.train_cfgs.total_steps
                or truncated
            ):
                # state_tensor = torch.as_tensor(
                #     action_info['state_vec'], device=self.device, dtype=torch.float32
                # )
                _, terminal_value, terminal_cost_value, _ = self._actor_critic.step(state)
                # terminal_value, terminal_cost_value = torch.unsqueeze(
                #     terminal_value, 0
                # ), torch.unsqueeze(terminal_cost_value, 0)
                self._policy_buf.finish_path(terminal_value, terminal_cost_value)



