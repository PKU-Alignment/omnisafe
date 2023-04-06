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
"""Implementation of the SafeLOOP algorithm."""

from copy import deepcopy

import numpy as np
import torch
from gymnasium.utils.save_video import save_video

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.planner import ARCPlanner
from omnisafe.algorithms.model_based.policy_gradient import PolicyGradientModelBased
from omnisafe.models.actor_q_critic import ActorQCritic
from omnisafe.utils import core
from omnisafe.utils.config_utils import namedtuple2dict
from omnisafe.wrappers.model_based_wrapper import ModelBasedEnvWrapper
from matplotlib import pylab
import os
@registry.register
class SafeLOOP(
    PolicyGradientModelBased, ARCPlanner
):  # pylint: disable=too-many-instance-attributes
    """The Safe Learning Off-Policy with Online Planning (SafeLOOP) algorithm.

    References:
        Title: Learning Off-Policy with Online Planning
        Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        URL: https://arxiv.org/abs/2008.10066
    """

    def __init__(self, env_id, cfgs) -> None:
        PolicyGradientModelBased.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        # # Initialize Actor-Critic
        self.actor_critic = self.set_algorithm_specific_actor_critic()
        self.ac_targ = deepcopy(self.actor_critic)
        self._ac_training_setup()

        self.alpha = self.cfgs.alpha
        self.alpha_gamma = self.cfgs.alpha_gamma
        ARCPlanner.__init__(
            self,
            self.algo,
            self.cfgs,
            self.device,
            self.env,
            self.virtual_env,
            self.actor_critic,
            **namedtuple2dict(self.cfgs.mpc_config),
        )

        # Set up model saving
        what_to_save = {
            'ac': self.actor_critic,
            'dynamics': self.dynamics,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()

    # pylint: disable-next=too-many-locals
    def compute_loss_v(self, data):
        """Computing value loss.

        Args:
            data (dict): data from replay buffer.

        Returns:
            torch.Tensor.
        """
        obs, act, rew, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
        q_value_list = self.actor_critic.critic(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, logp_a_next = self.ac_targ.actor.predict(
                obs, deterministic=False, need_log_prob=True
            )
            q_targ = torch.min(torch.vstack(self.ac_targ.critic(obs_next, act_targ)), dim=0).values
            backup = rew + self.cfgs.gamma * (1 - done) * (q_targ - self.alpha * logp_a_next)
        # MSE loss against Bellman backup
        loss_q = []
        q_values = []
        for q_value in q_value_list:
            loss_q.append(torch.mean((q_value - backup) ** 2))
            q_values.append(torch.mean(q_value))

        # Useful info for logging
        q_info = dict(QVals=sum(q_values).cpu().detach().numpy())
        return sum(loss_q), q_info

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Args:
            data (dict): data from replay buffer.

        Returns:
            torch.Tensor.
        """
        action, logp_a = self.actor_critic.actor.predict(
            data['obs'], deterministic=True, need_log_prob=True
        )
        loss_pi = self.actor_critic.critic(data['obs'], action)[0] - self.alpha * logp_a
        pi_info = {'LogPi': logp_a.cpu().detach().numpy()}
        return -loss_pi.mean(), pi_info

    def update_policy_net(self, data) -> None:
        """Update policy network.

        Args:
            data (dict): data dictionary.
        """
        # Train policy with one steps of gradient descent
        self.actor_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()
        self.logger.store(**{'Loss/Pi': loss_pi.item()})

    def alpha_discount(self):
        """Alpha discount."""
        self.alpha *= self.alpha_gamma

    def polyak_update_target(self):
        """Polyak update target network."""
        with torch.no_grad():
            for param, param_targ in zip(self.actor_critic.parameters(), self.ac_targ.parameters()):
                # Notes: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                param_targ.data.mul_(self.cfgs.polyak)
                param_targ.data.add_((1 - self.cfgs.polyak) * param.data)

    def update_value_net(self, data: dict) -> None:
        """Update value network.

        Args:
            data (dict): data dictionary
        """
        # Train value critic with one steps of gradient descent
        self.critic_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_v(data)
        loss_q.backward()
        self.critic_optimizer.step()
        self.logger.store(**{'Loss/Value': loss_q.item(), 'QVals': q_info['QVals']})

    def set_algorithm_specific_actor_critic(self):
        """
        Use this method to initialize network.
        e.g. Initialize Soft Actor Critic

        Returns:
            Actor_critic
        """
        self.actor_critic = ActorQCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            standardized_obs=self.cfgs.standardized_obs,
            shared_weights=self.cfgs.model_cfgs.shared_weights,
            model_cfgs=self.cfgs.model_cfgs,
            device=self.device,
        ).to(self.device)
        # Set up optimizer for policy and q-function
        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=self.cfgs.actor_lr
        )
        self.critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.critic, learning_rate=self.cfgs.critic_lr
        )
        return self.actor_critic

    def _ac_training_setup(self):
        """Set up target network for off_policy training."""
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for param in self.ac_targ.actor.parameters():
            param.requires_grad = False
        for param in self.ac_targ.critic.parameters():
            param.requires_grad = False

    def algorithm_specific_logs(self, time_step):
        """Log algo parameter"""
        super().algorithm_specific_logs(time_step)
        if time_step >= self.cfgs.update_policy_start_timesteps:
            self.logger.log_tabular('Loss/DynamicsTrainMseLoss')
            self.logger.log_tabular('Loss/DynamicsValMseLoss')
            self.logger.log_tabular('Plan/safety_costs_mean')
            self.logger.log_tabular('QVals')
            self.logger.log_tabular('Loss/Pi', std=False)
            self.logger.log_tabular('Loss/Value')
        else:
            self.logger.store(
                **{
                    'Loss/Pi': 0,
                    'Plan/safety_costs_mean': 0,
                    'QVals': 0,
                    'Loss/Value': 0,
                }
            )
            self.logger.log_tabular('QVals')
            self.logger.log_tabular('Loss/Pi', std=False)
            self.logger.log_tabular('Loss/Value')
            self.logger.log_tabular('Loss/DynamicsTrainMseLoss')
            self.logger.log_tabular('Loss/DynamicsValMseLoss')
            self.logger.log_tabular('Plan/safety_costs_mean')

    def update_actor_critic(self, time_step):
        """update actor and critic"""
        if time_step >= self.cfgs.update_policy_start_timesteps:
            for _ in range(self.cfgs.update_policy_iters):
                data = self.off_replay_buffer.sample_batch()
                # First run one gradient descent step for Q.
                self.update_value_net(data)

                # Freeze Q-network so you don't waste computational effort
                # computing gradients for it during the policy learning step.
                for param in self.actor_critic.critic.parameters():
                    param.requires_grad = False

                # Next run one gradient descent step for actor.
                self.update_policy_net(data)

                # Unfreeze Q-network so you can optimize it at next DDPG step.
                for param in self.actor_critic.critic.parameters():
                    param.requires_grad = True

                # Finally, update target networks by polyak averaging.
                self.polyak_update_target()
                self.alpha_discount()

    def update_dynamics_model(self):
        """Update dynamics."""
        state = self.off_replay_buffer.obs_buf[: self.off_replay_buffer.size, :]
        action = self.off_replay_buffer.act_buf[: self.off_replay_buffer.size, :]
        reward = self.off_replay_buffer.rew_buf[: self.off_replay_buffer.size]
        cost = self.off_replay_buffer.cost_buf[: self.off_replay_buffer.size]
        next_state = self.off_replay_buffer.obs_next_buf[: self.off_replay_buffer.size, :]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        if self.env.env_type == 'mujoco-velocity':
            labels = np.concatenate(
                (
                    np.reshape(reward, (reward.shape[0], -1)),
                    np.reshape(cost, (cost.shape[0], -1)),
                    delta_state,
                ),
                axis=-1,
            )
        elif self.env.env_type == 'gym':
            labels = np.concatenate(
                (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
            )
        train_mse_losses, val_mse_losses = self.dynamics.train(
            inputs, labels, batch_size=256, holdout_ratio=0.2
        )
        self.logger.store(
            **{
                'Loss/DynamicsTrainMseLoss': train_mse_losses,
                'Loss/DynamicsValMseLoss': val_mse_losses,
            }
        )

    def select_action(self, time_step, state, env):
        """action selection"""
        if time_step < self.cfgs.update_policy_start_timesteps:
            action = self.env.action_space.sample()

        else:
            action, safety_costs_mean = self.get_action(np.array(state))
            self.logger.store(
                **{
                    'Plan/safety_costs_mean': safety_costs_mean,
                }
            )
            action = action + np.random.normal(action.shape) * self.cfgs.exploration_noise
        action = np.clip(action, env.action_space.low, env.action_space.high)
        return action, None

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
    ):  # pylint: disable=too-many-arguments
        """store real data"""
        if not terminated and not truncated and not info['goal_met']:
            # Current goal position is not related to the last goal position, so do not store.
            self.off_replay_buffer.store(
                obs=state, act=action, rew=reward, cost=cost, next_obs=next_state, done=truncated
            )

    def algo_reset(self):
        """reset planner"""
        if self.env.env_type == 'gym':
            self.planner_reset()

    def draw_picture(self, save_replay_path, timestep, num_episode, dist_pred, dist_true, name='reward'):
        """draw a curve of the predicted value and the ground true value"""
        target1 = list(dist_pred)
        target2 = list(dist_true)
        input1 = np.arange(0, np.array(dist_pred).shape[0], 1)
        input2 = np.arange(0, np.array(dist_pred).shape[0], 1)

        pylab.plot(input1, target1, 'r-', label='pred')
        pylab.plot(input2, target2, 'b-', label='true')
        pylab.xlabel('Step')
        pylab.ylabel(name)
        pylab.xticks(np.arange(0, np.array(dist_pred).shape[0], 50))  # Set the axis numbers
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

    #pylint: disable-next=too-many-locals, too-many-statements
    def eval_policy_dymaics(self, eval_episode, timestep):
        """evaluate the policy and dynamics"""
        env_kwargs = {
            'render_mode': 'rgb_array',
            'camera_id': None,
            'camera_name': 'fixedfar',
        }
        env = ModelBasedEnvWrapper('SafeLOOP', self.env_id, **env_kwargs)

        env.env.reset(seed=timestep)
        env.set_eplen(int(self.cfgs.max_ep_len))

        obs,_ = env.reset()
        terminated, truncated = False, False
        ep_len, ep_ret, ep_cost = 0, 0, 0
        frames = []
        dist_pred, dist_true = [], []
        reward_pred, reward_true = [], []
        dist_min_pred, dist_min_true = [], []
        num_episode = 0
        while True:
            if terminated or truncated:
                print(f'Eval Episode Return: {ep_ret} \t Cost: {ep_cost}')
                save_replay_path = os.path.join(self.logger.log_dir,'video-pic')
                save_video(
                    frames,
                    save_replay_path,
                    fps=env.env.metadata['render_fps'],
                    episode_trigger=lambda x: True,
                    episode_index=timestep + num_episode,
                    name_prefix='eval',
                )
                self.draw_picture(save_replay_path, timestep, num_episode, dist_pred, dist_true, 'dist0')
                self.draw_picture(save_replay_path, timestep, num_episode, reward_pred, reward_true, 'reward')
                self.draw_picture(save_replay_path, timestep, num_episode, dist_min_pred, dist_min_true, 'dist_min')

                frames = []
                dist_pred, dist_true = [], []
                dist_min_pred, dist_min_true = [], []

                reward_pred, reward_true = [], []

                ep_len, ep_ret, ep_cost = 0, 0, 0
                obs, _ = env.reset()
                num_episode += 1
                if eval_episode == num_episode:
                    break
            action, safety_costs_mean = self.get_action(np.array(obs))

            #action, _ = self.planner.get_action(np.array(obs))
            curr_state = np.array([obs] * 1)
            #Shape: [1, state_dim]
            curr_state = np.expand_dims(curr_state, axis=0)
            #Shape: [1, 1, state_dim]
            curr_state = np.repeat(curr_state, self.cfgs.dynamics_cfgs.network_size, 0)
            #Shape: [network_size, 1, state_dim]
            state_traj = torch.from_numpy(curr_state).float().to(self.device)
            #Shape: [network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]
            # [ network_size, 1, state_dim]
            curr_action = np.array([action] * 1)
            #Shape: [1, action_dim]
            curr_action = np.expand_dims(curr_action, axis=0)
            curr_action = np.repeat(curr_action, self.cfgs.dynamics_cfgs.network_size, 0)
            action_traj = torch.from_numpy(curr_action).float().to(self.device)
            #use all dynamics model to predict next state (all_model=True)
            next_states, _ = self.virtual_env.safeloop_step(
                state_traj,
                action_traj,
                all_model=True,
                repeat_network=False,
            )
            #next_states and var shape:
            #[ network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]
            hazard_obs = next_states[0, 0, env.key_to_slice['hazards']].reshape(1, -1, 2).cpu().detach().numpy()
            #hazard_obs = next_states[0, 0, env.lidar_key_to_slice['hazards']].reshape(-1).cpu().detach().numpy()

            #select the first dynamics model result

            hazards_dist = np.sqrt(np.sum(np.square(hazard_obs), axis=2)).reshape(-1)
            cost_numpy = []

            for _, context in enumerate(hazards_dist):
                cost_numpy.append(context)

            obs, reward, cost, terminated, truncated, info = env.step(
                action, self.cfgs.action_repeat
            )
            dist_min_pred.append(np.min(np.array(cost_numpy)))
            dist_min_true.append(np.min(np.array(info['hazards_dist'])))
            dist_pred.append(np.array(cost_numpy)[0])
            dist_true.append(info['hazards_dist'][0])
            reward_pred.append(next_states[0, 0, 0].item())
            reward_true.append(reward)

            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            frames.append(env.render())
        env.close()