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

import numpy as np
import scipy.stats as stats
import torch
from torch import jit

class CCEPlanner():
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
                 cost_gamma,
                 cost_limit,
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
        self._cost_gamma = cost_gamma
        self._cost_limit = cost_limit
        self._device = device
        self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
        #self._action_sequence_var = ((action_max - action_min)**2)/16 * torch.ones(self._horizon, *self._action_shape, device=self._device)
        self._action_sequence_var = 2 * torch.ones(self._horizon, *self._action_shape, device=self._device)

    @torch.no_grad()
    def _act_from_last_gaus(self,state, last_mean, last_var):
        # Sample actions from the last gaussian distribution
        # Constrain the variance to be less than the distance to the boundary
        left_dist, right_dist = last_mean - self._action_min, self._action_max - last_mean
        #constrained_var = torch.minimum(torch.minimum(torch.square(left_dist / 2), torch.square(right_dist / 2)), last_var)
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

    @torch.no_grad()
    def _select_elites(self, actions, rewards, costs, values=None, cost_values=None):
        """
        Compute the return of the actions
        """
        assert actions.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
        assert rewards.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"
        assert costs.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"

        costs = costs.reshape(self._horizon, self._num_particles, self._num_samples, 1)
        sum_horizon_costs = torch.sum(costs, dim=0)
        mean_particles_costs = sum_horizon_costs.mean(dim=0)
        mean_episode_costs = mean_particles_costs * (1000/self._horizon)

        returns = rewards.reshape(self._horizon, self._num_particles,  self._num_samples, 1)
        sum_horizon_returns = torch.sum(returns, dim=0)
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        mean_episode_returns = mean_particles_returns * (1000/self._horizon)

        assert mean_particles_returns.shape[0] == self._num_samples

        feasible_num = torch.sum(mean_episode_costs <= self._cost_limit).item()
        if feasible_num < self._num_elites:
            elite_values, elite_actions = -mean_episode_costs, actions
        else:
            elite_idxs = (mean_episode_costs <= self._cost_limit).nonzero().reshape(-1) # like tensor([0, 1])
            elite_values, elite_actions = mean_episode_returns[elite_idxs], actions[:, elite_idxs]

        elite_idxs_topk = torch.topk(elite_values.squeeze(1), self._num_elites, dim=0).indices
        elite_returns_topk, elite_actions_topk = elite_values[elite_idxs_topk], elite_actions[:, elite_idxs_topk]
        info = {
            'Plan/feasible_num': feasible_num,
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
            'Plan/episode_costs_max': mean_episode_costs.max().item(),
            'Plan/episode_costs_mean': mean_episode_costs.mean().item(),
            'Plan/episode_costs_min': mean_episode_costs.min().item(),
        }

        return elite_returns_topk, elite_actions_topk, info

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
        while iter < self._num_iterations and last_var.max() > self._epsilon:
            actions = self._act_from_last_gaus(state, last_mean=last_mean, last_var=last_var)
            # [horizon, num_sample, action_shape]
            states_repeat, actions_repeat = self._state_action_repeat(state, actions)
            # [num_particles * num_samples/num_ensemble, state_shape], [horizon, num_particles * num_samples/num_ensemble, action_shape]
            traj = self._dynamics.imagine(states_repeat, self._horizon, actions_repeat)
            # {states, rewards, values}, each value shape is [horizon, num_ensemble, num_particles * num_samples/num_ensemble, 1]

            elite_values, elite_actions, info = self._select_elites(actions, traj['rewards'], traj['costs'])
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