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
from omnisafe.algorithms.model_based.planner import ARCPlanner
import numpy as np
from matplotlib import pylab
from gymnasium.utils.save_video import save_video
import os
from torch import nn, optim


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class LOOP(BaseAlgo):
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
        assert int(self._cfgs.train_cfgs.total_steps) % self._cfgs.logger_cfgs.log_cycle == 0
        self._total_steps = int(self._cfgs.train_cfgs.total_steps)
        self._steps_per_epoch = int(self._cfgs.logger_cfgs.log_cycle)
        self._epochs = self._total_steps // self._cfgs.logger_cfgs.log_cycle
    def _init_model(self) -> None:
        self._dynamics_state_space = self._env.coordinate_observation_space if hasattr(self._env, 'coordinate_observation_space') else self._env.observation_space
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

        self._planner = ARCPlanner(
            dynamics=self._dynamics,
            num_models=self._cfgs.dynamics_cfgs.num_ensemble,
            horizon=self._cfgs.algo_cfgs.plan_horizon,
            num_iterations=self._cfgs.algo_cfgs.num_iterations,
            num_particles=self._cfgs.algo_cfgs.num_particles,
            num_samples=self._cfgs.algo_cfgs.num_samples,
            num_elites=self._cfgs.algo_cfgs.num_elites,
            momentum=self._cfgs.algo_cfgs.momentum,
            epsilon=self._cfgs.algo_cfgs.epsilon,
            gamma=self._cfgs.algo_cfgs.gamma,
            device=self._device,
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._env.action_space.shape,
            action_max=1.0,
            action_min=-1.0,
        )

        self._use_actor_critic = True
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)




    def _init(self) -> None:
        self._dynamics_buf = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.dynamics_cfgs.batch_size,
            device=self._device,
        )
        if self._cfgs.evaluation_cfgs.use_eval:
            self._eval_fn = self._evaluation_single_step
        else:
            self._eval_fn = None

        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha], lr=self._cfgs.model_cfgs.critic.lr
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device)
            )

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

        self._logger.register_key('Value/alpha')
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.register_key('Loss/alpha_loss')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

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

    def algo_reset(self):
        pass

    def _update(self) -> None:
        for step in range(self._steps_per_sample // self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if step % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)

            self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs, next_action
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, target_q_value_r) + nn.functional.mse_loss(
            q2_value_r, target_q_value_r
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            }
        )


    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update cost critic using TD3 algorithm.

        Args:
            obs (torch.Tensor): current observation
            act (torch.Tensor): current action
            cost (torch.Tensor): current cost
            done (torch.Tensor): current done signal
            next_obs (torch.Tensor): next observation

        Returns:
            None
        """
        with torch.no_grad():
            # set the update noise and noise clip.
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_c, next_q2_value_c = self._actor_critic.target_cost_critic(
                next_obs, next_action
            )
            next_q_value_c = torch.max(next_q1_value_c, next_q2_value_c) - next_logp * self._alpha
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c

        q1_value_c, q2_value_c = self._actor_critic.cost_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_c, target_q_value_c) + nn.functional.mse_loss(
            q2_value_c, target_q_value_c
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        self._actor_critic.cost_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q1_value_c.mean().item(),
            }
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_pi': loss.mean().item(),
            }
        )
        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                **{
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                }
            )
        self._logger.store(
            **{
                'Value/alpha': self._alpha,
            }
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()
        return loss


    def _log_when_not_update(self) -> None:
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
                'Value/alpha': self._alpha,
            }
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                **{
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                }
            )
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store(
                **{
                    'Loss/alpha_loss': 0.0,
                }
            )

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
            self._dynamics_buf.store(
                obs=state, act=action, reward=reward, cost=cost, next_obs=next_state, done=done
            )

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

