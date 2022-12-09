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
from torch.nn.functional import softplus

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.aux import dist_xy, generate_lidar, get_reward_cost
from omnisafe.algorithms.model_based.policy_gradient import PolicyGradientModelBased


@registry.register
class MBPPOLag(PolicyGradientModelBased):
    """MBPPO-Lag"""

    def __init__(self, algo='mbppo-lag', clip=0.2, **cfgs):
        super().__init__(algo=algo, **cfgs)
        self.clip = clip
        self.cost_limit = self.cfgs['lagrange_cfgs']['cost_limit']
        self.device = torch.device(self.cfgs['device'])

        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0

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
        self.logger.log_tabular('Loss/Cost')
        self.logger.log_tabular('Loss/DeltaCost')
        self.logger.log_tabular('Penalty', softplus(self.penalty_param))
        self.logger.log_tabular('Values/V')
        self.logger.log_tabular('Values/C')
        self.logger.log_tabular('Megaiter')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('PolicyRatio')

    def update_actor_critic(self):
        """update actor critic"""
        # -------------------train actor and critic ----------------------
        megaiter = 0
        perf_flag = True
        while perf_flag:
            if megaiter == 0:
                last_valid_rets = [0] * 6

            # env_model2 = torch.load(exp_name+"env_model.pkl")
            # predict_env2 = PredictEnv(self.algo, self.dynamics, self.exp_name, 'pytorch')

            self.roll_out_in_imaginary(
                megaiter,
                self.predict_env,
                30000,
                penalty_param=0,
                cost_criteria=self.cost_criteria,
                use_cost_critic=self.cfgs['use_cost_critic'],
                gamma=self.cfgs['On_buffer_cfgs']['gamma'],
            )

            # ---------------validation--------------------------------------
            if megaiter > 0:
                old_params_pi = self.get_param_values(self.actor_critic.pi)
                old_params_v = self.get_param_values(self.actor_critic.v)
                old_params_vc = self.get_param_values(self.actor_critic.vc)
                self.update()
                result, valid_rets = self.mbppo_valid(
                    last_valid_rets,
                    self.predict_env,
                    penalty_param=0,
                    cost_criteria=self.cost_criteria,
                    use_cost_critic=self.cfgs['use_cost_critic'],
                    gamma=self.cfgs['On_buffer_cfgs']['gamma'],
                )
                if result is True:
                    perf_flag = False
                    # ------------backtarck-----------------
                    self.set_param_values(old_params_pi, self.actor_critic.pi)
                    self.set_param_values(old_params_v, self.actor_critic.v)
                    self.set_param_values(old_params_vc, self.actor_critic.vc)
                    megaiter += 1
                    break

                megaiter += 1
                last_valid_rets = valid_rets

            else:
                megaiter += 1
                self.update()

        self.logger.store(Megaiter=megaiter)

    def compute_loss_v(self, data):
        """compute the loss of value function"""
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        obs.to(self.device)
        ret.to(self.device)
        cret.to(self.device)
        return ((self.actor_critic.v(obs) - ret) ** 2).mean(), (
            (self.actor_critic.vc(obs) - cret) ** 2
        ).mean()

    def compute_loss_pi(self, data):
        """compute the loss of policy"""
        dist, _log_p = self.actor_critic.pi(data['obs'], data['act'].to(self.device))
        ratio = torch.exp(_log_p - data['logp'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        # loss_pi -= self.entropy_coef * dist.entropy().mean()

        p = softplus(self.penalty_param)
        penalty_item = p.item()
        loss_pi += penalty_item * ((ratio * data['cadv']).mean())
        loss_pi /= 1 + penalty_item

        # Useful extra info
        approx_kl = (data['logp'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip) | ratio.lt(1 - self.clip)
        clipfrac = torch.as_tensor(clipped, device=device, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def update_dynamics_model(self):
        """compute the loss of dynamics"""
        state, action, _, _, next_state, _ = self.env_pool.sample(len(self.env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = delta_state
        self.predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

    def update(self):
        """update ac"""
        data = self.buf.get()
        cur_cost = self.logger.get_stats('DynaMetrics/EpCost')[0]
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        beta_safety = self.cfgs['lagrange_cfgs']['beta']
        cost_dev = cur_cost - self.cost_limit * beta_safety
        loss_penalty = -self.penalty_param * cost_dev
        self.penalty_optimizer.zero_grad()
        loss_penalty.backward()
        self.penalty_optimizer.step()
        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = self.compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item()

        # Train policy with multiple steps of gradient descent
        train_pi_iters = 80
        for i in range(train_pi_iters):

            loss_pi, pi_info = self.compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])

            kl = pi_info['kl']
            if kl > 1.2 * 0.01:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)
            self.pi_optimizer.step()

        # Value function learning
        train_v_iters = 80
        for i in range(train_v_iters):
            loss_v, loss_vc = self.compute_loss_v(data)
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            # mpi_avg_grads(ac.v)   # average grads across MPI processes
            self.vf_optimizer.step()
            self.cvf_optimizer.zero_grad()
            loss_vc.backward()
            # mpi_avg_grads(ac.vc)  # average grads across MPI processes
            self.cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(
            **{
                'Loss/Pi': pi_l_old,
                'Loss/Value': v_l_old,
                'Loss/Cost': cv_l_old,
                'Loss/DeltaPi': (loss_pi.item() - pi_l_old),
                'Loss/DeltaValue': (loss_v.item() - v_l_old),
                'Loss/DeltaCost': (loss_vc.item() - cv_l_old),
                'Entropy': ent,
                'KL': kl,
                'PolicyRatio': cf,
            }
        )

    def get_param_values(self, model):
        """get the dynamics parameters"""
        trainable_params = list(model.parameters())  # + [self.log_std]
        params = np.concatenate([p.contiguous().view(-1).data.numpy() for p in trainable_params])
        return params.copy()

    def set_param_values(self, new_params, model, set_new=True):
        """set the dynamics parameters"""
        trainable_params = list(model.parameters())

        param_shapes = [p.data.numpy().shape for p in trainable_params]
        # print("param shapes",len(param_shapes))
        param_sizes = [p.data.numpy().size for p in trainable_params]
        if set_new:
            current_idx = 0
            for idx, param in enumerate(trainable_params):
                vals = new_params[current_idx : current_idx + param_sizes[idx]]
                vals = vals.reshape(param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += param_sizes[idx]

    # collect experience in env to train dynamics models
    def roll_out_in_imaginary(
        self,
        megaiter,
        predict_env,
        local_steps_per_epoch,
        penalty_param,
        cost_criteria,
        use_cost_critic,
        gamma,
    ):
        """collect data and store to experience buffer."""
        max_ep_len2 = 80
        o, static = self.env.reset()  ##generate the initial state!!!!
        o = np.clip(o, -1000, 1000)
        goal_pos = static['goal']  ### generate the initial goal position!!!!
        hazards_pos = static['hazards']  ### generate the initial hazards position!!!!
        ld = dist_xy(o[40:], goal_pos)  ## the distance between robot and goal
        dep_ret = 0
        dep_cost = 0
        dep_len = 0
        # print("training policy with imagination")
        if megaiter == 0:
            mix_real = 1500
        else:
            mix_real = 0

        for t in range(local_steps_per_epoch - mix_real):

            # generate hazard lidar
            obs_vec = generate_lidar(o, hazards_pos)
            robot_pos = o[40:]
            obs_vec = np.array(obs_vec)
            obs_vec = np.clip(obs_vec, -1000, 1000)

            otensor = torch.as_tensor(obs_vec, device=device, dtype=torch.float32)
            a, v, vc, logp = self.actor_critic.step(otensor)
            del otensor

            if True in np.isnan(a):
                print('produce nan in actor')
                print('action,obs', a, obs_vec)
                a = np.nan_to_num(a)
            a = np.clip(a, self.env.action_space.low, self.env.action_space.high)

            # --------USING LEARNED MODEL OF ENVIRONMENT TO GENERATE ROLLOUTS-----------------
            next_o = predict_env.step(o, a)

            if True in np.isnan(next_o):
                print('produce nan in actor')
                print('next_o,action,obs', next_o, a, o)
                next_o = np.nan_to_num(next_o)
            next_o = np.clip(next_o, -1000, 1000)
            r, c, ld, goal_flag = get_reward_cost(ld, robot_pos, hazards_pos, goal_pos)

            dep_ret += r
            dep_cost += c
            dep_len += 1

            # save and log
            self.buf.store(obs=obs_vec, act=a, rew=r, crew=c, val=v, cval=vc, logp=logp)

            # Update obs (critical!)
            o = next_o

            # Model horizon (H)  = max_ep_len2
            timeout = dep_len == max_ep_len2
            terminal = timeout
            epoch_ended = t == local_steps_per_epoch - 1
            if terminal or epoch_ended or goal_flag:
                # if epoch_ended and not(terminal):
                #     print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended or goal_flag:
                    otensort = torch.as_tensor(obs_vec, device=device, dtype=torch.float32)
                    _, v, vc, _ = self.actor_critic.step(otensort)
                    del otensort
                else:
                    v = 0
                    vc = 0
                self.buf.finish_path(v, vc)
                # self.buf.finish_path(v, vc, penalty_param=float(0))
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(
                        **{
                            'DynaMetrics/EpRet': dep_ret,
                            'DynaMetrics/EpLen': dep_len + 1,
                            'DynaMetrics/EpCost': dep_cost,
                        }
                    )

                o, static = self.env.reset()
                o = np.clip(o, -1000, 1000)
                goal_pos = static['goal']
                hazards_pos = static['hazards']
                ld = dist_xy(o[40:], goal_pos)
                dep_ret, dep_len, dep_cost = 0, 0, 0

    def mbppo_valid(
        self, last_valid_rets, predict_env, penalty_param, cost_criteria, use_cost_critic, gamma
    ):
        """validation whether improve policy"""
        # 6 ELITE MODELS OUT OF 8
        valid_rets = [0] * 6
        winner = 0
        # print("validating............")
        for va in range(len(valid_rets)):
            ov, staticv = self.env.reset()  ##########  create initial state!!!!!
            ov = np.clip(ov, -1000, 1000)

            goal_posv = staticv['goal']
            hazards_posv = staticv['hazards']
            ldv = dist_xy(ov[40:], goal_posv)
            step_iter = 0
            while step_iter < 75:
                obs_vecv = generate_lidar(ov, hazards_posv)  ## just generate lidar observation??
                robot_posv = ov[40:]  ## use true robot position
                obs_vecv = np.array(obs_vecv)
                obs_vecv = np.clip(obs_vecv, -1000, 1000)

                ovt = torch.as_tensor(obs_vecv, dtype=torch.float32, device=device)
                av, _, _, _ = self.actor_critic.step(ovt)
                if True in np.isnan(av):
                    print('produce nan in vali actor')
                    print('action,obs', av, obs_vecv)
                    av = np.nan_to_num(av)
                av = np.clip(av, self.env.action_space.low, self.env.action_space.high)
                del ovt
                next_ov = predict_env.step_elite(ov, av, va)
                if True in np.isnan(next_ov):
                    print('produce nan in  vali env')
                    print('next_o,action,obs', next_ov, av, ov)
                    next_ov = np.nan_to_num(next_ov)
                next_ov = np.clip(next_ov, -1000, 1000)
                rv, cv, ldv, goal_flagv = get_reward_cost(ldv, robot_posv, hazards_posv, goal_posv)
                valid_rets[va] += rv
                ov = next_ov
                if goal_flagv:
                    ov, staticv = self.env.reset()
                    ov = np.clip(ov, -1000, 1000)
                    goal_posv = staticv['goal']
                    hazards_posv = staticv['hazards']
                    ldv = dist_xy(ov[40:], goal_posv)
                step_iter += 1
            if valid_rets[va] > last_valid_rets[va]:
                winner += 1
        # print(valid_rets,last_valid_rets)
        performance_ratio = winner / 6
        # print("Performance ratio=",performance_ratio)
        thresh = 4 / 6  # BETTER THAN 50%

        return performance_ratio < thresh, valid_rets
