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


import torch
from torch import nn

from omnisafe.adapter import ModelBasedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger

from omnisafe.algorithms.model_based.models import EnsembleDynamicsModel
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.common.buffer import VectorOnPolicyBuffer


from omnisafe.algorithms.model_based.planner import CEMPlanner
import numpy as np
from matplotlib import pylab
from gymnasium.utils.save_video import save_video
import os


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class MBPPO(BaseAlgo):
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
            use_truncated=False,
            use_var=False,
            use_reward_critic=False,
            use_cost_critic=False,
            actor_critic=None,
            rew_func=None,
            cost_func=None,
            truncated_func=None,
        )
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

        self._use_actor_critic = True
        self._policy_state_space = self._env.lidar_observation_space if self._env.lidar_observation_space is not None else self._env.observation_space
        self._actor_critic = ConstraintActorCritic(
            obs_space=self._policy_state_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        # Set up optimizer for policy and value function


    def _init(self) -> None:
        self._dynamics_buf = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.dynamics_cfgs.batch_size,
            device=self._device,
        )


        self._policy_buf = VectorOnPolicyBuffer(
            obs_space=self._policy_state_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
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


        self.logger.register_key('VirtualMetrics/EpRet')
        self.logger.register_key('VirtualMetrics/EpLen')
        self.logger.register_key('VirtualMetrics/EpCost')
        self.logger.register_key('Loss/DynamicsTrainMseLoss')
        self.logger.register_key('Loss/DynamicsValMseLoss')
        self.logger.register_key('Loss/Pi')
        self.logger.register_key('Loss/Value')
        self.logger.register_key('Loss/DeltaPi')
        self.logger.register_key('Loss/DeltaValue')
        self.logger.register_key('Loss/CValue')
        self.logger.register_key('Loss/DeltaCValue')
        self.logger.register_key('Penalty')
        self.logger.register_key('Values/Adv')
        self.logger.register_key('Values/Adv_C')
        self.logger.register_key('Megaiter')
        self.logger.register_key('Entropy')
        self.logger.register_key('KL')
        self.logger.register_key('Misc/StopIter')
        self.logger.register_key('PolicyRatio')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/UpdateDynamics')
        if self._use_actor_critic:
            self._logger.register_key('Time/UpdateActorCritic')
        if self._cfgs.evaluation_cfgs.use_eval:
            self._logger.register_key('Time/Eval')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def learn(self) -> Tuple[Union[int, float], ...]:
        """This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        current_step = 0
        for epoch in range(self._epochs):
            current_step = self._env.roll_out(
                current_step=current_step,
                roll_out_step=self._steps_per_epoch,
                use_actor_critic=False,
                act_func=self._select_action,
                store_data_func=self.store_real_data,
                update_dynamics_func=self.update_dynamics_model,
                eval_func=self._eval_fn,
                logger=self._logger,
                algo_reset_func=None,
                update_actor_func=None,
                )
            # Evaluate episode
            self._logger.store(
                **{
                    'Train/Epoch': epoch,
                    'TotalEnvSteps': current_step,
                    'Time/Total': time.time() - start_time,
                }
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

    def select_action(self, time_step, state, env):  # pylint: disable=unused-argument
        """
        Select action when interact with real environment.

        Returns:
                action, action_info
        """
        if self._env.env_type == 'gym':
            state = self._env.generate_lidar(state)
        act, value_r, value_c, logp =  self._actor_critic.step(state)

        action_info = {'actor_state': state, 'val': value_r, 'cval': value_c, 'logp': logp}
        return act, action_info

    def algo_reset(self):
        pass

    def imagine_rollout(self):
        if initial_states is None:
            initial_states = random_choice(self.replay_buffer.get('states'), size=self.rollout_batch_size)
        buffer = self._create_buffer(self.rollout_batch_size * self.horizon)
        states = initial_states
        for t in range(self.horizon):
            with torch.no_grad():
                actions = policy.act(states, eval=False)
                next_states, rewards = self.model_ensemble.sample(states, actions)
            dones = self.check_done(next_states)
            violations = self.check_violation(next_states)
            buffer.extend(states=states, actions=actions, next_states=next_states,
                          rewards=rewards, dones=dones, violations=violations)
            continues = ~(dones | violations)
            if continues.sum() == 0:
                break
            states = next_states[continues]

        self.virt_buffer.extend(**buffer.get(as_dict=True))
        return buffer

    def update_actor_critic(self, time_step):  # pylint: disable=unused-argument
        """update actor critic"""
        megaiter = 0
        last_valid_rets = np.zeros(self.cfgs.dynamics_cfgs.elite_size)
        while True:
            self.imagine_rollout(megaiter)
            # validation
            if megaiter > 0:
                old_actor = self.get_param_values(self.actor_critic.actor)
                old_reward_critic = self.get_param_values(self.actor_critic.reward_critic)
                old_cost_critic = self.get_param_values(self.actor_critic.cost_critic)
                data = self.buf.get()
                ep_costs = self.logger.get_stats('DynaMetrics/EpCost')[0]
                self.update_lagrange_multiplier(ep_costs)
                self.update_policy_net(data=data)
                self.update_value_net(data=data)
                result, valid_rets = self.validation(last_valid_rets)
                if result is True:
                    # backtrack
                    self.set_param_values(old_actor, self.actor_critic.actor)
                    self.set_param_values(old_reward_critic, self.actor_critic.reward_critic)
                    self.set_param_values(old_cost_critic, self.actor_critic.cost_critic)
                    megaiter += 1
                    break
                megaiter += 1
                last_valid_rets = valid_rets
            else:
                megaiter += 1
                data = self.buf.get()
                ep_costs = self.logger.get_stats('DynaMetrics/EpCost')[0]
                self.update_lagrange_multiplier(ep_costs)
                self.update_policy_net(data=data)
                self.update_value_net(data=data)

        self.logger.store(Megaiter=megaiter)

    def compute_loss_v(self, data):
        """compute the loss of value function"""
        obs, ret, cret = data['obs'], data['target_value_r'], data['target_value_c']
        return ((self.actor_critic.reward_critic(obs) - ret) ** 2).mean(), (
            (self.actor_critic.cost_critic(obs) - cret) ** 2
        ).mean()
    def compute_loss_pi(self, data):
        """compute the loss of policy"""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['logp'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv_r'], ratio_clip * data['adv_r'])).mean()

        # ensure that Lagrange multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += penalty * ((ratio * data['adv_c']).mean())
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = (data['logp'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip) | ratio.lt(1 - self.clip)
        clipfrac = torch.as_tensor(clipped, device=self.device, dtype=torch.float32).mean().item()
        pi_info = {'kl': approx_kl, 'ent': ent, 'cf': clipfrac}
        return loss_pi, pi_info

    def update_dynamics_model(self, current_step):
        """Update dynamics."""
        state = self._dynamics_buf.data['obs'][: self._dynamics_buf.size, :]
        action = self._dynamics_buf.data['act'][: self._dynamics_buf.size, :]
        reward = self._dynamics_buf.data['reward'][: self._dynamics_buf.size]
        cost = self._dynamics_buf.data['cost'][: self._dynamics_buf.size]
        next_state = self._dynamics_buf.data['next_obs'][: self._dynamics_buf.size, :]
        delta_state = next_state - state
        inputs = torch.cat((state, action), -1)
        inputs = torch.reshape(inputs, (inputs.shape[0], -1))

        labels = torch.cat(
            (
                torch.reshape(reward, (reward.shape[0], -1)),
                torch.reshape(delta_state,(delta_state.shape[0], -1))
            ),
            -1
        )
        inputs = inputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        train_mse_losses, val_mse_losses = self._dynamics.train(
            inputs, labels, holdout_ratio=0.2
        )
        # ep_costs = self._logger.get_stats('Metrics/EpCost')[0]
        # #update Lagrange multiplier parameter
        # self.update_lagrange_multiplier(ep_costs)
        self._logger.store(
            **{
                'Loss/DynamicsTrainMseLoss': train_mse_losses.item(),
                'Loss/DynamicsValMseLoss': val_mse_losses.item(),
            }
        )

    def update_policy_net(self, data):
        """update policy"""
        # Get prob. distribution before updates: used to measure KL distance
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs.actor_iters):
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl_div = pi_info['kl']
            if self.cfgs.kl_early_stopping:
                if kl_div > self.cfgs.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break
            self.actor_optimizer.zero_grad()
            loss_pi.backward()
            self.actor_optimizer.step()
        self.logger.store(
            **{
                'Loss/Pi': self.loss_pi_before,
                'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
                'Misc/StopIter': i + 1,
                'Values/Adv': data['adv_r'].cpu().numpy(),
                'Values/Adv_C': data['adv_c'].cpu().numpy(),
                'Entropy': pi_info_old['ent'],
                'KL': pi_info['kl'],
                'PolicyRatio': pi_info['cf'],
            }
        )

    def update_value_net(self, data):
        """Value function learning"""
        v_l_old, cv_l_old = self.compute_loss_v(data)
        self.loss_v_before, self.loss_c_before = v_l_old.item(), cv_l_old.item()

        for _ in range(self.cfgs.critic_iters):
            loss_v, loss_vc = self.compute_loss_v(data)
            self.reward_critic_optimizer.zero_grad()
            loss_v.backward()
            self.reward_critic_optimizer.step()

            self.cost_critic_optimizer.zero_grad()
            loss_vc.backward()
            self.cost_critic_optimizer.step()

        self.logger.store(
            **{
                'Loss/DeltaValue': loss_v.item() - self.loss_v_before,
                'Loss/Value': self.loss_v_before,
                'Loss/DeltaCValue': loss_vc.item() - self.loss_c_before,
                'Loss/CValue': self.loss_c_before,
            }
        )

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
                param.data = torch.from_numpy(vals).float().to(self.device)
                current_idx += param_sizes[idx]

    def imagine_rollout(self, megaiter):  # pylint: disable=too-many-locals
        """collect data and store to experience buffer."""
        state = self.env_auxiliary.reset()
        dep_ret, dep_cost, dep_len = 0, 0, 0
        mix_real = self.cfgs.mixed_real_time_steps if megaiter == 0 else 0

        for time_step in range(self.cfgs.imaging_steps_per_policy_update - mix_real):
            raw_action, action, action_info = self.select_action(time_step, state, self.env_auxiliary)
            next_state, reward, cost, info = self.virtual_step(state, action)

            dep_ret += reward
            dep_cost += (self.cost_gamma**dep_len) * cost
            dep_len += 1

            obs, reward, cost = expand_dims(
                action_info['state_vec'], reward, cost
            )
            self.buf.store(
                obs=torch.tensor(obs,device=self.cfgs.device),
                act=raw_action,
                reward=torch.tensor(reward,device=self.cfgs.device),
                value_r=action_info['val'].unsqueeze(0),
                logp=action_info['logp'],
                cost=torch.tensor(cost,device=self.cfgs.device),
                value_c=action_info['cval'].unsqueeze(0),
            )
            state = next_state

            timeout = dep_len == self.cfgs.horizon
            truncated = timeout
            epoch_ended = time_step == self.cfgs.imaging_steps_per_policy_update - 1
            if truncated or epoch_ended or info['goal_flag']:
                if timeout or epoch_ended or info['goal_flag']:
                    state_tensor = torch.as_tensor(
                        action_info['state_vec'], device=self.device, dtype=torch.float32
                    )
                    _, _, terminal_value, terminal_cost_value, _ = self.actor_critic.step(state_tensor)
                    del state_tensor
                    terminal_value, terminal_cost_value = torch.unsqueeze(
                        terminal_value, 0
                    ), torch.unsqueeze(terminal_cost_value, 0)
                else:
                    # this means episode is terminated,
                    # and this will be triggered only in robots fall down case
                    terminal_value, terminal_cost_value = torch.zeros(
                    1, dtype=torch.float32, device=self.cfgs.device
                ), torch.zeros(1, dtype=torch.float32, device=self.cfgs.device)

                self.buf.finish_path(terminal_value, terminal_cost_value)

                if timeout:
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(
                        **{
                            'DynaMetrics/EpRet': dep_ret,
                            'DynaMetrics/EpLen': dep_len,
                            'DynaMetrics/EpCost': dep_cost,
                        }
                    )
                state = self.env_auxiliary.reset()
                dep_ret, dep_len, dep_cost = 0, 0, 0

    def validation(self, last_valid_rets):
        """policy validation"""
        valid_rets = np.zeros(self.cfgs.validation_num)
        winner = 0
        for valid_id in range(len(valid_rets)):  # pylint:disable=consider-using-enumerate
            state = self.env_auxiliary.reset()
            for step in range(self.cfgs.validation_horizon):
                raw_action, action, _ = self.select_action(step, state, self.env_auxiliary)
                next_state, reward, _, info = self.virtual_step(state, action, idx=valid_id)
                valid_rets[valid_id] += reward
                state = next_state
                if info['goal_flag']:
                    state = self.env_auxiliary.reset()
            if valid_rets[valid_id] > last_valid_rets[valid_id]:
                winner += 1
        performance_ratio = winner / self.cfgs.validation_num
        threshold = self.cfgs.validation_threshold_num / self.cfgs.validation_num
        result = performance_ratio < threshold
        return result, valid_rets

    def _select_action(
            self,
            current_step: int,
            state: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """action selection"""
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
            #action = torch.rand(size=1, *self._env.action_space.shape)
        else:
            action = self._planner.output_action(state)
            #action = action.cpu().detach().numpy()
        assert action.shape == torch.Size([state.shape[0], self._env.action_space.shape[0]]), "action shape should be [batch_size, action_dim]"
        info = {}
        return action, info

    def store_real_data(
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
    ) -> None:  # pylint: disable=too-many-arguments
        """Store real data in buffer."""
        done = terminated or truncated
        if 'goal_met' not in info.keys():
            goal_met = False
        else:
            goal_met = info['goal_met']
        if not terminated and not truncated and not goal_met:
            # if goal_met == true, Current goal position is not related to the last goal position, this huge transition will confuse the dynamics model.
            self._true_buf.store(
                obs=state, act=action, reward=reward, cost=cost, next_obs=next_state, done=done
            )


    # pylint: disable-next=too-many-arguments
    def store_real_data(
        self,
        time_step,
        ep_len,
        state,
        action_info,
        action,
        raw_action,
        reward,
        cost,
        terminated,
        truncated,
        next_state,
        info,
    ):
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

        """store real data"""
        if not terminated and not truncated and not info['goal_met']:
            self.off_replay_buffer.store(
                obs=torch.tensor(state, device=self.device),
                act=torch.tensor(action, device=self.device),
                reward=torch.tensor(reward, device=self.device),
                cost=torch.tensor(cost, device=self.device),
                next_obs=torch.tensor(next_state, device=self.device),
                done=torch.tensor(truncated, device=self.device),
            )
        if (
            current_step % self.cfgs.update_policy_freq <= self.cfgs.mixed_real_time_steps
            and self.buf.buffers[0].ptr < self.cfgs.mixed_real_time_steps
        ):
            obs, reward, cost = expand_dims(
                action_info['state_vec'], reward, cost
            )
            #print(torch.tensor(obs,device=self.cfgs.device),raw_action,torch.tensor(reward,device=self.cfgs.device),action_info['val'].unsqueeze(0),action_info['logp'],torch.tensor(cost,device=self.cfgs.device),action_info['cval'].unsqueeze(0))
            self.buf.store(
                obs=torch.tensor(obs,device=self.cfgs.device),
                act=raw_action,
                reward=torch.tensor(reward,device=self.cfgs.device),
                value_r=action_info['val'].unsqueeze(0),
                logp=action_info['logp'],
                cost=torch.tensor(cost,device=self.cfgs.device),
                value_c=action_info['cval'].unsqueeze(0),
            )
            if terminated:
                # this means episode is terminated,
                # which will be triggered only in robots fall down case
                terminal_value, terminal_cost_value = torch.zeros(
                1, dtype=torch.float32, device=self.cfgs.device
            ), torch.zeros(1, dtype=torch.float32, device=self.cfgs.device)

                self.buf.finish_path(terminal_value, terminal_cost_value)

            # reached max imaging horizon, mixed real timestep, real max timestep , or episode truncated.
            elif (
                current_step % self.cfgs.horizon < self.cfgs.action_repeat
                or self.buf.buffers[0].ptr == self.cfgs.mixed_real_time_steps
                or current_step >= self.cfgs.max_real_time_steps
                or truncated
            ):
                state_tensor = torch.as_tensor(
                    action_info['state_vec'], device=self.device, dtype=torch.float32
                )
                _, _, terminal_value, terminal_cost_value, _ = self.actor_critic.step(state_tensor)
                del state_tensor
                terminal_value, terminal_cost_value = torch.unsqueeze(
                    terminal_value, 0
                ), torch.unsqueeze(terminal_cost_value, 0)
                self.buf.finish_path(terminal_value, terminal_cost_value)


    def _evaluation_single_step(
            self,
            current_step: int,
    ) -> None:

        env_kwargs = {
            'render_mode': 'rgb_array',
            'camera_name': 'track',
        }
        eval_env = ModelBasedAdapter(
                    self._env_id, 1, self._seed, self._cfgs, **env_kwargs
                )
        obs,_ = eval_env.reset()
        terminated, truncated = False, False
        ep_len, ep_ret, ep_cost = 0, 0, 0
        frames = []
        obs_pred, obs_true = [], []
        reward_pred, reward_true = [], []
        num_episode = 0
        while True:
            if terminated or truncated:
                print(f'Eval Episode Return: {ep_ret} \t Cost: {ep_cost}')
                save_replay_path = os.path.join(self._logger.log_dir,'video-pic')
                self._logger.store(
                    **{
                        'EvalMetrics/EpRet': ep_ret.item(),
                        'EvalMetrics/EpCost': ep_cost.item(),
                        'EvalMetrics/EpLen': ep_len,
                    }
                )
                save_video(
                    frames,
                    save_replay_path,
                    fps=30,
                    episode_trigger=lambda x: True,
                    episode_index=current_step + num_episode,
                    name_prefix='eval',
                )
                self.draw_picture(
                    timestep=current_step,
                    num_episode=self._cfgs.evaluation_cfgs.num_episode,
                    pred_state=obs_pred,
                    true_state=obs_true,
                    save_replay_path=save_replay_path,
                    name='obs_mean'
                )
                self.draw_picture(
                    timestep=current_step,
                    num_episode=self._cfgs.evaluation_cfgs.num_episode,
                    pred_state=reward_pred,
                    true_state=reward_true,
                    save_replay_path=save_replay_path,
                    name='reward'
                )
                frames = []
                obs_pred, obs_true = [], []

                reward_pred, reward_true = [], []

                ep_len, ep_ret, ep_cost = 0, 0, 0
                obs, _ = eval_env.reset()
                num_episode += 1
                if num_episode == self._cfgs.evaluation_cfgs.num_episode:
                    break
            action, _ = self._select_action(current_step, obs)

            idx = np.random.choice(self._dynamics.elite_model_idxes, size=1)
            traj = self._dynamics.imagine(states=obs, horizon=1, idx=idx, actions=action.unsqueeze(0))

            pred_next_obs_mean = traj['states'][0][0].mean()
            pred_reward = traj['rewards'][0][0]

            obs, reward, cost, terminated, truncated, info = eval_env.step(action)
            true_next_obs_mean = obs.mean()

            obs_pred.append(pred_next_obs_mean.item())
            obs_true.append(true_next_obs_mean.item())

            reward_pred.append(pred_reward.item())
            reward_true.append(reward.item())

            ep_ret += reward
            ep_cost += cost
            ep_len += info['num_step']
            frames.append(eval_env.render())

    def draw_picture(
        self,
        timestep: int,
        num_episode: int,
        pred_state: list,
        true_state: list,
        save_replay_path: str="./",
        name: str='reward'
        ) -> None:
        """draw a curve of the predicted value and the ground true value"""
        target1 = list(pred_state)
        target2 = list(true_state)
        input1 = np.arange(0, np.array(pred_state).shape[0], 1)
        input2 = np.arange(0, np.array(pred_state).shape[0], 1)

        pylab.plot(input1, target1, 'r-', label='pred')
        pylab.plot(input2, target2, 'b-', label='true')
        pylab.xlabel('Step')
        pylab.ylabel(name)
        pylab.xticks(np.arange(0, np.array(pred_state).shape[0], 50))  # Set the axis numbers
        if name == 'reward':
            pylab.yticks(np.arange(0, 3, 0.2))
        else:
            pylab.yticks(np.arange(0, 1, 0.2))
        pylab.legend(
            loc=3, borderaxespad=2.0, bbox_to_anchor=(0.7, 0.7)
        )  # Sets the position of that box for what each line is
        pylab.grid()  # draw grid
        pylab.savefig(
            os.path.join(save_replay_path,
            str(name)
            + str(timestep)
            + '_'
            + str(num_episode)
            + '.png'),
            dpi=200,
        )  # save as picture
        pylab.close()

