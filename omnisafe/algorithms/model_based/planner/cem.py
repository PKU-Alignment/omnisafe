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

class CEMPlanner():
    def __init__(self,
                 dynamics,
                 num_models,
                 horizon,
                 num_iterations,
                 num_particles,
                 num_samples,
                 num_elites,
                 momentum,
                 epsilon,
                 gamma,
                 device,
                 dynamics_state_shape,
                 action_shape,
                 action_max,
                 action_min,
                 ) -> None:
        assert (num_samples * num_particles) % num_models == 0, "num_samples * num_elites should be divisible by num_models"
        assert num_particles % num_models == 0, "num_particles should be divisible by num_models"
        self._dynamics = dynamics
        self._num_models = num_models
        self._horizon = horizon
        self._num_iterations = num_iterations
        self._num_particles = num_particles
        self._num_samples = num_samples
        self._num_elites = num_elites
        self._momentum = momentum
        self._epsilon = epsilon
        self._dynamics_state_shape = dynamics_state_shape
        self._action_shape = action_shape
        self._action_max = action_max
        self._action_min = action_min
        self._gamma = gamma
        self._device = device
        self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
        self._action_sequence_var = 4 * torch.ones(self._horizon, *self._action_shape, device=self._device)

    @torch.no_grad()
    def _act_from_last_gaus(self, last_mean, last_var):
        constrained_std = torch.sqrt(last_var)
        actions = torch.clamp(last_mean.unsqueeze(1) + constrained_std.unsqueeze(1)  * \
            torch.randn(self._horizon, self._num_samples, *self._action_shape, device=self._device),self._action_min, self._action_max)
        actions.clamp_(min=self._action_min, max=self._action_max)  # Clip action range
        return actions

    @torch.no_grad()
    def _state_action_repeat(self, state, action):
        """
        Repeat the state for num_repeat * action.shape[0] times and action for num_repeat times
        """
        assert action.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._num_samples, self._action_shape)"
        assert state.shape == torch.Size([1, *self._dynamics_state_shape]) , "state dimension one should be 1"
        states = state.repeat(int(self._num_particles/self._num_models * self._num_samples), 1)
        actions = action.unsqueeze(1).repeat(1, int(self._num_particles/self._num_models), 1, 1)
        actions = actions.reshape(self._horizon, int(self._num_particles/self._num_models * self._num_samples), *self._action_shape)
        return states, actions

    # @torch.no_grad()
    # def _compute_actions_return(self, rewards, values=None):
    #     """
    #     Compute the return of the actions
    #     """
    #     assert rewards.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"
    #     returns = rewards.reshape(self._horizon, self._num_particles,  self._num_samples, 1)
    #     sum_horizon_returns = torch.sum(returns, dim=0)
    #     mean_particles_returns = sum_horizon_returns.mean(dim=0)

    #     assert mean_particles_returns.shape[0] == self._num_samples
    #     return mean_particles_returns
    # @torch.no_grad()
    # def _select_elites(self, actions, returns, num_elites):

    #     assert actions.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
    #     assert returns.shape == torch.Size([self._num_samples, 1]), "Input returns dimension should be equal to (self._num_samples, 1)"

    #     elite_idxs = torch.topk(returns.squeeze(1), self._num_elites, dim=0).indices
    #     elite_returns, elite_actions = returns[elite_idxs], actions[:, elite_idxs]

    #     info = {}
    #     info['elite_idxs'] = elite_idxs
    #     info['best_action'] = elite_actions[0,0].unsqueeze(0)
    #     assert info['best_action'].shape == torch.Size([1, *self._action_shape])

    #     return elite_actions, elite_returns, info

    @torch.no_grad()
    def _select_elites(self, actions, traj):
        rewards = traj['rewards']
        assert actions.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
        assert rewards.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"
        returns = rewards.reshape(self._horizon, self._num_particles,  self._num_samples, 1)
        sum_horizon_returns = torch.sum(returns, dim=0)
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        mean_episode_returns = mean_particles_returns * (1000/self._horizon)
        assert mean_episode_returns.shape == torch.Size([self._num_samples, 1]), "Input returns dimension should be equal to (self._num_samples, 1)"
        elite_idxs = torch.topk(mean_episode_returns.squeeze(1), self._num_elites, dim=0).indices
        elite_values, elite_actions = mean_episode_returns[elite_idxs], actions[:, elite_idxs]

        info = {
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
        }
        info['elite_idxs'] = elite_idxs
        info['best_action'] = elite_actions[0,0].unsqueeze(0)
        assert info['best_action'].shape == torch.Size([1, *self._action_shape])

        return elite_values, elite_actions, info

    @torch.no_grad()
    def _update_mean_var(self, elite_actions, elite_values):

        assert elite_actions.shape == torch.Size([self._horizon, self._num_elites, *self._action_shape]), "Input elite_actions dimension should be equal to (self._horizon, self._num_elites, self._action_shape)"
        assert elite_values.shape == torch.Size([self._num_elites, 1]), "Input elite_values dimension should be equal to (self._num_elites, 1)"

        new_mean = elite_actions.mean(dim=1)
        new_var = elite_actions.var(dim=1)

        return new_mean, new_var

    @torch.no_grad()
    def output_action(self,state):
        assert state.shape == torch.Size([1, *self._dynamics_state_shape]), "Input state dimension should be equal to (1, self._dynamics_state_shape)"
        last_mean = torch.zeros_like(self._action_sequence_mean)
        last_var = self._action_sequence_var.clone()
        last_mean[:-1] = self._action_sequence_mean[1:].clone()
        last_mean[-1] = self._action_sequence_mean[-1].clone()

        iter = 0
        while iter < self._num_iterations:
            actions = self._act_from_last_gaus(last_mean=last_mean, last_var=last_var)
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