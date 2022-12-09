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

import copy

import numpy as np
import torch
from torch.nn.functional import softplus

from omnisafe.algos import registry
from omnisafe.algos.common.buffer import Buffer
from omnisafe.algos.model_based.lagrange import Lagrange
from omnisafe.algos.model_based.policy_gradient import PolicyGradientModelBased


@registry.register
class MBPPOLag(PolicyGradientModelBased, Lagrange):
    """MBPPO-Lag"""

    def __init__(self, algo='mbppo-lag', clip=0.2, **cfgs):
        PolicyGradientModelBased.__init__(self, algo=algo, **cfgs)
        Lagrange.__init__(self, **self.cfgs['lagrange_cfgs'], device=self.cfgs['device'])
        self.clip = clip
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.env_auxiliary = copy.deepcopy(self.env)
        self.buf = Buffer(
            actor_critic=self.actor_critic,
            obs_dim=self.env.ac_state_size,
            act_dim=self.env.action_space.shape[0],
            scale_rewards=self.cfgs['scale_rewards'],
            standardized_obs=self.cfgs['standardized_obs'],
            size=self.cfgs['imaging_steps_per_policy_update'],
            **self.cfgs['buffer_cfgs'],
            device=self.device,
        )

    def algorithm_specific_logs(self, timestep):
        """log algo parameter"""
        super().algorithm_specific_logs(timestep)
        self.logger.log_tabular('DynaMetrics/EpRet')
        self.logger.log_tabular('DynaMetrics/EpLen')
        self.logger.log_tabular('DynaMetrics/EpCost')
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        self.logger.log_tabular('Loss/CValue')
        self.logger.log_tabular('Loss/DeltaCValue')
        self.logger.log_tabular('Penalty', softplus(self.lagrangian_multiplier))
        self.logger.log_tabular('Values/Adv')
        self.logger.log_tabular('Values/Adv_C')
        self.logger.log_tabular('Megaiter')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('PolicyRatio')

    def update_actor_critic(self, timestep):  # pylint: disable=unused-argument
        """update actor critic"""
        megaiter = 0
        last_valid_rets = np.zeros(self.cfgs['dynamics_cfgs']['elite_size'])
        while True:
            self.roll_out_in_imaginary(megaiter)
            # validation
            if megaiter > 0:
                old_params_pi = self.get_param_values(self.actor_critic.pi)
                old_params_v = self.get_param_values(self.actor_critic.v)
                old_params_vc = self.get_param_values(self.actor_critic.vc)
                data = self.buf.get()
                cur_cost = self.logger.get_stats('DynaMetrics/EpCost')[0]
                self.update_lagrange_multiplier(cur_cost)
                self.update_policy_net(data=data)
                self.update_value_net(data=data)
                result, valid_rets = self.validation(last_valid_rets)
                if result is True:
                    # backtrack
                    self.set_param_values(old_params_pi, self.actor_critic.pi)
                    self.set_param_values(old_params_v, self.actor_critic.v)
                    self.set_param_values(old_params_vc, self.actor_critic.vc)
                    megaiter += 1
                    break
                megaiter += 1
                last_valid_rets = valid_rets
            else:
                megaiter += 1
                data = self.buf.get()
                cur_cost = self.logger.get_stats('DynaMetrics/EpCost')[0]
                self.update_lagrange_multiplier(cur_cost)
                self.update_policy_net(data=data)
                self.update_value_net(data=data)
        self.logger.store(Megaiter=megaiter)

    def compute_loss_v(self, data):
        """compute the loss of value function"""
        obs, ret, cret = data['obs'], data['target_v'], data['target_c']
        return ((self.actor_critic.v(obs) - ret) ** 2).mean(), (
            (self.actor_critic.vc(obs) - cret) ** 2
        ).mean()

    def compute_loss_pi(self, data):
        """compute the loss of policy"""
        dist, _log_p = self.actor_critic.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        penalty = softplus(self.lagrangian_multiplier)
        penalty_item = penalty.item()
        loss_pi += penalty_item * ((ratio * data['cost_adv']).mean())
        loss_pi /= 1 + penalty_item
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
        next_state = self.off_replay_buffer.obs2_buf[: self.off_replay_buffer.size, :]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = delta_state
        self.virtual_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

    def update_policy_net(self, data):
        """update policy"""
        # Get prob. distribution before updates: used to measure KL distance
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs['pi_iters']):
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl_div = pi_info['kl']
            if self.cfgs.get('kl_early_stopping', False):
                if kl_div > 1.2 * self.cfgs['target_kl']:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()
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

        for _ in range(self.cfgs['critic_iters']):
            loss_v, loss_vc = self.compute_loss_v(data)
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            self.vf_optimizer.step()

            self.cvf_optimizer.zero_grad()
            loss_vc.backward()
            self.cvf_optimizer.step()

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
        mix_real = self.cfgs['mixed_real_time_steps'] if megaiter == 0 else 0

        for time_step in range(self.cfgs['imaging_steps_per_policy_update'] - mix_real):
            action, action_info = self.select_action(time_step, state, self.env_auxiliary)
            next_state = self.virtual_env.step(state, action)
            next_state = np.nan_to_num(next_state)
            next_state = np.clip(next_state, -self.cfgs['obs_clip'], self.cfgs['obs_clip'])
            reward, cost, goal_flag = self.env_auxiliary.get_reward_cost(next_state)

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

            timeout = dep_len == self.cfgs['horizon']
            truncated = timeout
            epoch_ended = time_step == self.cfgs['imaging_steps_per_policy_update'] - 1
            if truncated or epoch_ended or goal_flag:
                if timeout or epoch_ended or goal_flag:
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
        valid_rets = np.zeros(self.cfgs['validation_num'])
        winner = 0
        for valid_id in range(len(valid_rets)):  # pylint:disable=consider-using-enumerate
            state = self.env_auxiliary.reset()
            for step in range(self.cfgs['validation_horizon']):
                action, _ = self.select_action(step, state, self.env_auxiliary)
                next_state = self.virtual_env.step_elite(state, action, valid_id)
                next_state = np.nan_to_num(next_state)
                next_state = np.clip(next_state, -self.cfgs['obs_clip'], self.cfgs['obs_clip'])
                reward, _, goal_flag = self.env_auxiliary.get_reward_cost(next_state)
                valid_rets[valid_id] += reward
                state = next_state
                if goal_flag:
                    state = self.env_auxiliary.reset()
            if valid_rets[valid_id] > last_valid_rets[valid_id]:
                winner += 1
        performance_ratio = winner / self.cfgs['validation_num']
        threshold = self.cfgs['validation_threshold_num'] / self.cfgs['validation_num']
        result = performance_ratio < threshold
        return result, valid_rets

    # pylint: disable-next=too-many-arguments
    def store_real_data(
        self,
        timestep,
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
            timestep % self.cfgs['update_policy_freq'] <= self.cfgs['mixed_real_time_steps']
            and self.buf.ptr < self.cfgs['mixed_real_time_steps']
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
                timestep % self.cfgs['horizon'] < self.cfgs['action_repeat']
                or self.buf.ptr == self.cfgs['mixed_real_time_steps']
                or timestep >= self.cfgs['max_real_time_steps']
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
