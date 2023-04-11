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
"""Safe controllers which do a black box optimization incorporating the constraint costs."""

import torch

class ARCPlanner():
    def __init__(self,
                 dynamics,
                 actor_critic,
                 num_models,
                 horizon,
                 num_iterations,
                 num_particles,
                 num_samples,
                 num_elites,
                 mixture_coefficient,
                 temperature,
                 momentum,
                 epsilon,
                 gamma,
                 device,
                 dynamics_state_shape,
                 action_shape,
                 action_max,
                 action_min,
                 ) -> None:
        self._dynamics = dynamics
        self._actor_critic = actor_critic
        self._num_models = num_models
        self._horizon = horizon
        self._num_iterations = num_iterations
        self._num_particles = num_particles
        self._num_samples = num_samples
        self._num_elites = num_elites
        self._mixture_coefficient = mixture_coefficient
        self._actor_traj = int(self._mixture_coefficient * self._num_samples)
        self._num_action = self._actor_traj + self._num_samples
        self._temperature = temperature
        self._momentum = momentum
        self._epsilon = epsilon
        self._dynamics_state_shape = dynamics_state_shape
        self._action_shape = action_shape
        self._action_max = action_max
        self._action_min = action_min
        self._gamma = gamma
        self._device = device
        self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
        #self._action_sequence_var = ((action_max - action_min)**2)/16 * torch.ones(self._horizon, *self._action_shape, device=self._device)
        self._action_sequence_var = 4 * torch.ones(self._horizon, *self._action_shape, device=self._device)
    @torch.no_grad()
    def _act_from_last_gaus(self,state, last_mean, last_var):
        # Sample actions from the last gaussian distribution
        # Constrain the variance to be less than the distance to the boundary
        left_dist, right_dist = last_mean - self._action_min, self._action_max - last_mean
        #constrained_var = torch.minimum(torch.minimum(torch.square(left_dist / 2), torch.square(right_dist / 2)), last_var)
        constrained_std = torch.sqrt(last_var)
        actions = torch.clamp(last_mean.unsqueeze(1) + constrained_std.unsqueeze(1)  * \
            torch.randn(self._horizon, self._num_samples, *self._action_shape, device=self._device),self._action_min, self._action_max)

        return actions
    @torch.no_grad()
    def _act_from_actor(self, state):
        assert state.shape == torch.Size([1, *self._dynamics_state_shape]) , "state dimension one should be 1"
        assert self._actor_traj % self._num_models == 0, "actor_traj should be divisible by num_models"
        traj = self._dynamics.imagine(states=state, horizon=self._horizon, actions=None, actor_critic=self._actor_critic, idx=0)
        actions = traj['actions'].reshape(self._horizon, 1, *self._action_shape).clone().repeat([1, self._actor_traj, 1])
        return actions
    @torch.no_grad()
    def _state_action_repeat(self, state, action):
        """
        Repeat the state for num_repeat * action.shape[0] times and action for num_repeat times
        """
        assert action.shape == torch.Size([self._horizon, self._num_action, *self._action_shape]), "Input action dimension should be equal to (self._num_samples, self._action_shape)"
        assert state.shape == torch.Size([1, *self._dynamics_state_shape]) , "state dimension one should be 1"
        states = state.repeat(int(self._num_particles * self._num_action), 1)
        actions = action.unsqueeze(1).repeat(1, int(self._num_particles), 1, 1)
        actions = actions.reshape(self._horizon, int(self._num_particles * self._num_action), *self._action_shape)
        return states, actions

    @torch.no_grad()
    def _select_elites(self, actions, traj):
        """
        Compute the return of the actions
        """
        rewards = traj['rewards']
        values = traj['values']
        assert actions.shape == torch.Size([self._horizon, self._num_action, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
        assert rewards.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles*self._num_action), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles*self._num_samples, 1)"
        assert values.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles*self._num_action), 1]), "Input values dimension should be equal to (self._horizon, self._num_models, self._num_particles*self._num_samples, 1)"

        rewards = rewards.reshape(self._horizon, self._num_models*self._num_particles,  self._num_action, 1)
        values = values.reshape(self._horizon, self._num_models*self._num_particles,  self._num_action, 1)
        sum_horizon_returns = torch.sum(rewards, dim=0) + values[-1,:,:,:]
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        mean_episode_returns = mean_particles_returns * (1000/self._horizon)

        assert mean_episode_returns.shape[0] == self._num_action


        elite_actions = actions
        elite_values = mean_episode_returns
        info = {
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
        }

        return elite_values, elite_actions, info


    @torch.no_grad()
    def _update_mean_var(self, elite_actions, elite_values):

        assert elite_actions.shape[0] == self._horizon and elite_actions.shape[-1] == self._action_shape[0], "Input elite_actions dimension should be equal to (self._horizon, self._num_elites, self._action_shape)"
        assert elite_values.shape[-1] == 1, "Input elite_values dimension should be equal to (self._num_elites, 1)"
        assert elite_actions.shape[1] == elite_values.shape[0], "Number of action should be the same"

        max_value = elite_values.max(0)[0]
        score = torch.exp(self._temperature*(elite_values - max_value))
        score /= score.sum(0)
        new_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
        new_var = torch.sum(score.unsqueeze(0) * (elite_actions - new_mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)
        new_var = new_var.clamp_(0, 2)

        return new_mean, new_var

    @torch.no_grad()
    def output_action(self,state):
        assert state.shape == torch.Size([1, *self._dynamics_state_shape]), "Input state dimension should be equal to (1, self._dynamics_state_shape)"
        last_mean = torch.zeros_like(self._action_sequence_mean)
        last_var = self._action_sequence_var.clone()
        last_mean[:-1] = self._action_sequence_mean[1:].clone()
        last_mean[-1] = self._action_sequence_mean[-1].clone()

        iter = 0
        actions_actor = self._act_from_actor(state)
        while iter < self._num_iterations and last_var.max() > self._epsilon:
            actions_gauss = self._act_from_last_gaus(state, last_mean=last_mean, last_var=last_var)
            actions = torch.cat([actions_gauss, actions_actor], dim=1)
            # [horizon, num_sample, action_shape]
            states_repeat, actions_repeat = self._state_action_repeat(state, actions)
            # [num_particles * num_samples/num_ensemble, state_shape], [horizon, num_particles * num_samples/num_ensemble, action_shape]
            traj = self._dynamics.imagine(states_repeat, self._horizon, actions_repeat)
            # {states, rewards, values}, each value shape is [horizon, num_ensemble, num_particles * num_samples/num_ensemble, 1]

            elite_values, elite_actions, info = self._select_elites(actions, traj)
            # [num_sample, 1]
            new_mean, new_var = self._update_mean_var(elite_actions, elite_values)
            last_mean = self._momentum * last_mean + (1 - self._momentum) * new_mean
            last_var = self._momentum * last_var + (1 - self._momentum) * new_var
            iter += 1
        logger_info = {
            'Plan/iter': iter,
            'Plan/last_var_mean': last_var.mean().item(),
            'Plan/last_var_max': last_var.max().item(),
            'Plan/last_var_min': last_var.min().item(),
        }
        logger_info.update(info)
        self._action_sequence_mean = last_mean.clone()
        return last_mean[0].clone().unsqueeze(0), logger_info

    # def reset_planner(self):
    #     self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
    #     self._action_sequence_std = 2 * torch.ones(self._horizon, *self._action_shape, device=self._device)