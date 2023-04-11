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
from omnisafe.algorithms.model_based.planner.cem import CEMPlanner
import numpy as np
from matplotlib import pylab
from gymnasium.utils.save_video import save_video
import os


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class PETS(BaseAlgo):
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

        self._planner = CEMPlanner(
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

        self._use_actor_critic = False
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

        self._logger.register_key('Plan/iter')
        self._logger.register_key('Plan/last_var_mean')
        self._logger.register_key('Plan/last_var_max')
        self._logger.register_key('Plan/last_var_min')
        self._logger.register_key('Plan/episode_returns_max')
        self._logger.register_key('Plan/episode_returns_mean')
        self._logger.register_key('Plan/episode_returns_min')

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
                use_actor_critic=self._use_actor_critic,
                act_func=self._select_action,
                store_data_func=self._store_real_data,
                update_dynamics_func=self._update_dynamics_model,
                eval_func=self._eval_fn,
                logger=self._logger,
                algo_reset_func=self._algo_reset,
                update_actor_func=self._update_policy,
                )
            if current_step > self._cfgs.algo_cfgs.start_learning_steps:
                # update something per epoch
                # e.g. update lagrange multiplier
                self._update_epoch()
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

    def _algo_reset(self, current_step):
        pass

    def _update_policy(self, current_step):
        pass
    def _update_dynamics_model(self, current_step):
        """Update dynamics."""
        state = self._dynamics_buf.data['obs'][: self._dynamics_buf.size, :]
        action = self._dynamics_buf.data['act'][: self._dynamics_buf.size, :]
        reward = self._dynamics_buf.data['reward'][: self._dynamics_buf.size]
        cost = self._dynamics_buf.data['cost'][: self._dynamics_buf.size]
        next_state = self._dynamics_buf.data['next_obs'][: self._dynamics_buf.size, :]
        delta_state = next_state - state
        inputs = torch.cat((state, action), -1)
        inputs = torch.reshape(inputs, (inputs.shape[0], -1))

        labels = torch.reshape(delta_state,(delta_state.shape[0], -1))
        if self._cfgs.dynamics_cfgs.predict_reward:
            labels = torch.cat(((torch.reshape(reward, (reward.shape[0], -1))),labels), -1)
        if self._cfgs.dynamics_cfgs.predict_cost:
            labels = torch.cat(((torch.reshape(cost, (cost.shape[0], -1))),labels), -1)

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
    def _update_epoch(self):
        pass

    def _select_action(
            self,
            current_step: int,
            state: torch.Tensor,
            env: ModelBasedAdapter,
            ) -> Tuple[np.ndarray, Dict]:
        """action selection"""
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
            #action = torch.rand(size=1, *self._env.action_space.shape)
        else:
            action, info = self._planner.output_action(state)
            #action = action.cpu().detach().numpy()
            self._logger.store(
                **{
                'Plan/iter': info['Plan/iter'],
                'Plan/last_var_max': info['Plan/last_var_max'],
                'Plan/last_var_mean': info['Plan/last_var_mean'],
                'Plan/last_var_min': info['Plan/last_var_min'],
                'Plan/episode_returns_max': info['Plan/episode_returns_max'],
                'Plan/episode_returns_mean': info['Plan/episode_returns_mean'],
                'Plan/episode_returns_min': info['Plan/episode_returns_min'],
                }
            )
        assert action.shape == torch.Size([state.shape[0], self._env.action_space.shape[0]]), "action shape should be [batch_size, action_dim]"
        info = {}
        return action, info

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
            use_real_input: bool=True,
    ) -> None:

        env_kwargs = {
            'render_mode': 'rgb_array',
            'camera_name': 'track',
        }
        eval_env = ModelBasedAdapter(
                    self._env_id, 1, self._seed, self._cfgs, **env_kwargs
                )
        obs,_ = eval_env.reset()
        obs_dynamics = obs
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
            action, _ = self._select_action(current_step, obs, eval_env)

            idx = np.random.choice(self._dynamics.elite_model_idxes, size=1)
            traj = self._dynamics.imagine(states=obs_dynamics, horizon=1, idx=idx, actions=action.unsqueeze(0))

            pred_next_obs_mean = traj['states'][0][0].mean()
            pred_reward = traj['rewards'][0][0]

            obs, reward, cost, terminated, truncated, info = eval_env.step(action)

            obs_dynamics = obs if use_real_input else traj['states'][0][0]

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

