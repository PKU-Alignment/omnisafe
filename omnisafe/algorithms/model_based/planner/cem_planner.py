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


class CEMPlanner:
    def __init__(self,
                 dynamics,
                 num_models,
                 horizon,
                 num_iterations,
                 num_particles,
                 num_samples,
                 num_elites,
                 momentum,
                 gamma,
                 device,
                 state_shape,
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
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._action_max = action_max
        self._action_min = action_min
        self._gamma = gamma
        self._device = device
        self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
        self._action_sequence_std = 2 * torch.ones(self._horizon, *self._action_shape, device=self._device)

    def _generate_action_from_policy(self,state):
        actions = torch.clamp(self._action_sequence_mean.unsqueeze(1) + self._action_sequence_std.unsqueeze(1)  * \
            torch.randn(self._horizon, self._num_samples, *self._action_shape, device=self._device),self._action_min, self._action_max)

        return actions

    def _state_action_repeat(self, state, action):
        """
        Repeat the state for num_repeat * action.shape[0] times and action for num_repeat times
        """
        assert action.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._num_samples, self._action_shape)"
        assert state.shape == torch.Size([1, *self._state_shape]) , "state dimension one should be 1"
        states = state.repeat(int(self._num_particles/self._num_models * self._num_samples), 1)
        actions = action.unsqueeze(1).repeat(1, int(self._num_particles/self._num_models), 1, 1)
        actions = actions.reshape(self._horizon, int(self._num_particles/self._num_models * self._num_samples), *self._action_shape)
        return states, actions


    def _compute_actions_return(self, rewards, values=None):
        """
        Compute the return of the actions
        """
        #assert actions.shape == (self._horizon, self._num_samples, self._action_shape), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"

        assert rewards.shape[0:2] == torch.Size([self._horizon, self._num_models]) and rewards.shape[3] == 1
        # [horizon, num_particles * num_samples, 1]

        returns = rewards.reshape(self._horizon, self._num_samples, self._num_particles, 1)
        mean_particles_returns = returns.mean(dim=2)
        mean_horizon_returns = mean_particles_returns.mean(dim=0)

        assert mean_horizon_returns.shape[0] == self._num_samples
        return mean_horizon_returns

    def _select_elites(self, actions, returns, num_elites):

        assert actions.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
        assert returns.shape == torch.Size([self._num_samples, 1]), "Input returns dimension should be equal to (self._num_samples, 1)"

        elite_idxs = torch.topk(returns.squeeze(1), self._num_elites, dim=0).indices
        elite_returns, elite_actions = returns[elite_idxs], actions[:, elite_idxs]

        info = {}
        info['elite_idxs'] = elite_idxs
        info['best_action'] = elite_actions[0,0]
        return elite_actions, elite_returns, info

    def _update_mean_std(self, elite_actions, elite_values):

        assert elite_actions.shape == torch.Size([self._horizon, self._num_elites, *self._action_shape]), "Input elite_actions dimension should be equal to (self._horizon, self._num_elites, self._action_shape)"
        assert elite_values.shape == torch.Size([self._num_elites, 1]), "Input elite_values dimension should be equal to (self._num_elites, 1)"

        new_mean = elite_actions.mean(dim=1)
        new_std = elite_actions.std(dim=1)

        return new_mean, new_std

    def output_action(self,state):
        assert state.shape == torch.Size([1, *self._state_shape]), "Input state dimension should be equal to (1, self._state_shape)"
        for iter in range(self._num_iterations):
            actions = self._generate_action_from_policy(state)
            # [horizon, num_sample, action_shape]
            states_repeat, actions_repeat = self._state_action_repeat(state, actions)
            # [num_particles * num_samples/num_ensemble, state_shape], [horizon, num_particles * num_samples/num_ensemble, action_shape]
            traj = self._dynamics.imagine(states_repeat, self._horizon, actions_repeat)
            # {states, rewards, values}, each value shape is [horizon, num_ensemble, num_particles * num_samples/num_ensemble, 1]

            returns = self._compute_actions_return(traj['rewards'])
            # [num_sample, 1]
            elite_actions, elite_values, info = self._select_elites(actions=actions, returns=returns, num_elites=self._num_elites)
            new_mean, new_std = self._update_mean_std(elite_actions, elite_values)
            self._action_sequence_mean = self._momentum * self._action_sequence_mean + (1 - self._momentum) * new_mean
            self._action_sequence_std = self._momentum * self._action_sequence_std + (1 - self._momentum) * new_std
        return info['best_action']

    def reset_planner(self):
        self._action_sequence_mean = torch.zeros(self._horizon, *self._action_shape, device=self._device)
        self._action_sequence_std = 2 * torch.ones(self._horizon, *self._action_shape, device=self._device)