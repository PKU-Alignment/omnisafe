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
from omnisafe.algorithms.model_based.planner.cce import CCEPlanner

class CAPPlanner(CCEPlanner):
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
                 lagrange,
                 device,
                 dynamics_state_shape,
                 action_shape,
                 action_max,
                 action_min,
                 ) -> None:
        super().__init__(
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
        )
        self._lagrange = lagrange

    @torch.no_grad()
    def _select_elites(self, actions, traj):
        """
        Compute the return of the actions
        """
        rewards = traj['rewards']
        costs = traj['costs']
        vars = traj['vars']
        assert actions.shape == torch.Size([self._horizon, self._num_samples, *self._action_shape]), "Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)"
        assert rewards.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"
        assert vars.shape[:-1] == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples)]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, dynamics_state_shape)"
        assert costs.shape == torch.Size([self._horizon, self._num_models, int(self._num_particles/self._num_models*self._num_samples), 1]), "Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)"


        # var: [_horizon, network_size, num_gaussian_traj*particles/network_size, state_dim]
        var_penalty = vars.sqrt().norm(dim=3).max(1)[0]
        # cost_penalty: [_horizon, num_gaussian_traj*particles/network_size]
        var_penalty = var_penalty.repeat_interleave(self._num_models).view(
            costs.shape
        )
        # cost_penalty: [_horizon, num_gaussian_traj*particle]
        costs += self._lagrange.lagrangian_multiplier * var_penalty

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


