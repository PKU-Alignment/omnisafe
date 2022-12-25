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
"""MBPPOLag"""


import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.policy_gradient import PolicyGradientModelBased
from omnisafe.common.buffer import Buffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import core
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.wrappers import wrapper_registry


@registry.register
# pylint: disable-next=too-many-instance-attributes
class MBPPOLag(PolicyGradientModelBased, Lagrange):
    """The Model-based PPO-Lag algorithm.

    References:
        Title: Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm
        Authors: Ashish Kumar Jayant, Shalabh Bhatnagar
        URL: https://arxiv.org/abs/2210.07573
    """

    def __init__(self, env_id, cfgs) -> None:
        PolicyGradientModelBased.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        Lagrange.__init__(self, **namedtuple2dict(self.cfgs.lagrange_cfgs), device=self.cfgs.device)
        self.clip = self.cfgs.clip
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.env_auxiliary = wrapper_registry.get(self.wrapper_type)(self.algo, self.env_id)
        # Initialize Actor-Critic
        self.actor_critic = self.set_algorithm_specific_actor_critic()
        self.buf = Buffer(
            actor_critic=self.actor_critic,
            obs_dim=self.env.ac_state_size,
            act_dim=self.env.action_space.shape[0],
            scale_rewards=self.cfgs.scale_rewards,
            standardized_obs=self.cfgs.standardized_obs,
            size=self.cfgs.imaging_steps_per_policy_update,
            **namedtuple2dict(self.cfgs.buffer_cfgs),
            device=self.device,
        )
        # Set up model saving
        what_to_save = {
            'pi': self.actor_critic.actor,
            'dynamics': self.dynamics,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()

    def algorithm_specific_logs(self, time_step):
        """log algo parameter"""
        super().algorithm_specific_logs(time_step)
        self.logger.log_tabular('DynaMetrics/EpRet')
        self.logger.log_tabular('DynaMetrics/EpLen')
        self.logger.log_tabular('DynaMetrics/EpCost')
        self.logger.log_tabular('Loss/DynamicsTrainMseLoss')
        self.logger.log_tabular('Loss/DynamicsValMseLoss')
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        self.logger.log_tabular('Loss/CValue')
        self.logger.log_tabular('Loss/DeltaCValue')
        self.logger.log_tabular(
            'Penalty', self.lambda_range_projection(self.lagrangian_multiplier).item()
        )
        self.logger.log_tabular('Values/Adv')
        self.logger.log_tabular('Values/Adv_C')
        self.logger.log_tabular('Megaiter')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('PolicyRatio')

    def update_actor_critic(self, time_step):  # pylint: disable=unused-argument
        """update actor critic"""
        megaiter = 0
        last_valid_rets = np.zeros(self.cfgs.dynamics_cfgs.elite_size)
        while True:
            self.roll_out_in_imaginary(megaiter)
            # validation
            if megaiter > 0:
                old_actor = self.get_param_values(self.actor_critic.actor)
                old_reward_critic = self.get_param_values(self.actor_critic.reward_critic)
                old_cost_critic = self.get_param_values(self.actor_critic.cost_critic)
                self.update()
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
                self.update()

        self.logger.store(Megaiter=megaiter)

    def update(self):
        """Get data from buffer and update Lagrange multiplier, actor, critic"""
        data = self.buf.get()
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('DynaMetrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)

    def compute_loss_v(self, data):
        """compute the loss of value function"""
        obs, ret, cret = data['obs'], data['target_v'], data['target_c']
        return ((self.actor_critic.reward_critic(obs) - ret) ** 2).mean(), (
            (self.actor_critic.cost_critic(obs) - cret) ** 2
        ).mean()

    def compute_loss_pi(self, data):
        """compute the loss of policy"""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()

        # ensure that Lagrange multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += penalty * ((ratio * data['cost_adv']).mean())
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip) | ratio.lt(1 - self.clip)
        clipfrac = torch.as_tensor(clipped, device=self.device, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def update_dynamics_model(self):
        """compute the loss of dynamics"""
        state = self.off_replay_buffer.obs_buf[: self.off_replay_buffer.size, :]
        action = self.off_replay_buffer.act_buf[: self.off_replay_buffer.size, :]
        next_state = self.off_replay_buffer.obs_next_buf[: self.off_replay_buffer.size, :]
        reward = self.off_replay_buffer.rew_buf[: self.off_replay_buffer.size]
        cost = self.off_replay_buffer.cost_buf[: self.off_replay_buffer.size]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        if self.env.env_type == 'mujoco-speed':
            labels = np.concatenate(
                (
                    np.reshape(reward, (reward.shape[0], -1)),
                    np.reshape(cost, (cost.shape[0], -1)),
                    delta_state,
                ),
                axis=-1,
            )
        elif self.env.env_type == 'gym':
            labels = delta_state
        train_mse_losses, val_mse_losses = self.dynamics.train(
            inputs, labels, batch_size=256, holdout_ratio=0.2
        )
        self.logger.store(
            **{
                'Loss/DynamicsTrainMseLoss': train_mse_losses,
                'Loss/DynamicsValMseLoss': val_mse_losses,
            }
        )

    def update_policy_net(self, data):
        """update policy"""
        # Get prob. distribution before updates: used to measure KL distance
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs.pi_iters):
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
                'Values/Adv': data['adv'].cpu().numpy(),
                'Values/Adv_C': data['cost_adv'].cpu().numpy(),
                'Entropy': pi_info_old['ent'],
                'KL': pi_info['kl'],
                'PolicyRatio': pi_info['cf'],
            }
        )

    def update_value_net(self, data):
        """Value function learning"""
        v_l_old, cv_l_old = self.compute_loss_v(data)
        self.loss_v_before, self.loss_c_before, = (
            v_l_old.item(),
            cv_l_old.item(),
        )

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

    def roll_out_in_imaginary(self, megaiter):  # pylint: disable=too-many-locals
        """collect data and store to experience buffer."""
        state = self.env_auxiliary.reset()
        dep_ret, dep_cost, dep_len = 0, 0, 0
        mix_real = self.cfgs.mixed_real_time_steps if megaiter == 0 else 0

        for time_step in range(self.cfgs.imaging_steps_per_policy_update - mix_real):
            action, action_info = self.select_action(time_step, state, self.env_auxiliary)
            next_state, reward, cost, info = self.virtual_step(state, action)

            dep_ret += reward
            dep_cost += (self.cost_gamma**dep_len) * cost
            dep_len += 1

            self.buf.store(
                obs=action_info['state_vec'],
                act=action,
                rew=reward,
                val=action_info['val'],
                logp=action_info['logp'],
                cost=cost,
                cost_val=action_info['cval'],
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
                    _, val, cval, _ = self.actor_critic.step(state_tensor)
                    del state_tensor
                else:
                    # this means episode is terminated,
                    # and this will be triggered only in robots fall down case
                    val = 0
                    cval = 0
                self.buf.finish_path(val, cval, penalty_param=float(0))
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
                action, _ = self.select_action(step, state, self.env_auxiliary)
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

    # pylint: disable-next=too-many-arguments
    def store_real_data(
        self,
        time_step,
        ep_len,
        state,
        action_info,
        action,
        reward,
        cost,
        terminated,
        truncated,
        next_state,
        info,
    ):
        """store real data"""
        if not terminated and not truncated and not info['goal_met']:
            self.off_replay_buffer.store(
                obs=state, act=action, rew=reward, cost=cost, next_obs=next_state, done=truncated
            )
        if (
            time_step % self.cfgs.update_policy_freq <= self.cfgs.mixed_real_time_steps
            and self.buf.ptr < self.cfgs.mixed_real_time_steps
        ):
            self.buf.store(
                obs=action_info['state_vec'],
                act=action,
                rew=reward,
                val=action_info['val'],
                logp=action_info['logp'],
                cost=cost,
                cost_val=action_info['cval'],
            )
            if terminated:
                # this means episode is terminated,
                # which will be triggered only in robots fall down case
                val = 0
                cval = 0
                self.buf.finish_path(val, cval, penalty_param=float(0))

            # reached max imaging horizon, mixed real timestep, real max timestep , or episode truncated.
            elif (
                time_step % self.cfgs.horizon < self.cfgs.action_repeat
                or self.buf.ptr == self.cfgs.mixed_real_time_steps
                or time_step >= self.cfgs.max_real_time_steps
                or truncated
            ):
                state_tensor = torch.as_tensor(
                    action_info['state_vec'], device=self.device, dtype=torch.float32
                )
                _, val, cval, _ = self.actor_critic.step(state_tensor)
                del state_tensor
                self.buf.finish_path(val, cval, penalty_param=float(0))

    def algo_reset(self):
        """reset algo parameters"""

    def virtual_step(self, state, action, idx=None):
        """use virtual environment to predict next state, reward, cost"""
        if self.env.env_type == 'gym':
            next_state, _, _, _ = self.virtual_env.mbppo_step(state, action, idx)
            next_state = np.nan_to_num(next_state)
            next_state = np.clip(next_state, -self.cfgs.obs_clip, self.cfgs.obs_clip)
            reward, cost, goal_flag = self.env_auxiliary.get_reward_cost(next_state)
            info = {'goal_flag': goal_flag}
        elif self.env.env_type == 'mujoco-speed':
            next_state, reward, cost, _ = self.virtual_env.mbppo_step(state, action, idx)
            next_state = np.nan_to_num(next_state)
            reward = np.nan_to_num(reward)
            cost = np.nan_to_num(cost)
            next_state = np.clip(next_state, -self.cfgs.obs_clip, self.cfgs.obs_clip)
            info = {'goal_flag': False}
        return next_state, reward, cost, info

    def set_algorithm_specific_actor_critic(self):
        """
        Use this method to initialize network.
        e.g. Initialize Soft Actor Critic

        Returns:
            Actor_critic
        """
        self.actor_critic = ConstraintActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            scale_rewards=self.cfgs.scale_rewards,
            standardized_obs=self.cfgs.standardized_obs,
            model_cfgs=self.cfgs.model_cfgs,
        ).to(self.device)
        # Set up optimizer for policy and value function

        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=self.cfgs.actor_lr
        )
        self.reward_critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.reward_critic, learning_rate=self.cfgs.critic_lr
        )
        self.cost_critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.cost_critic, learning_rate=self.cfgs.critic_lr
        )

        return self.actor_critic
