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
"""Model Predictive Control Planner."""

import torch

from omnisafe.algorithms.model_based.planner.arc import ARCPlanner


class SafeARCPlanner(ARCPlanner):
    """The Safe Actor Regularized Control (ARC) trajectory optimization method.

    References:
        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `Safe ARC <https://arxiv.org/abs/2008.10066>`_
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self,
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
        cost_gamma,
        cost_limit,
        device,
        dynamics_state_shape,
        action_shape,
        action_max,
        action_min,
    ) -> None:
        super().__init__(
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
        )
        self._cost_gamma = cost_gamma
        self._cost_limit = cost_limit

    @torch.no_grad()
    def _select_elites(self, actions, traj):
        """
        Compute the return of the actions
        """
        rewards = traj['rewards']
        values = traj['values']
        costs = traj['costs']
        assert actions.shape == torch.Size(
            [self._horizon, self._num_action, *self._action_shape],
            # pylint: disable-next=line-too-long
        ), 'Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)'
        assert rewards.shape == torch.Size(
            [self._horizon, self._num_models, int(self._num_particles * self._num_action), 1],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles*self._num_samples, 1)'
        assert values.shape == torch.Size(
            [self._horizon, self._num_models, int(self._num_particles * self._num_action), 1],
            # pylint: disable-next=line-too-long
        ), 'Input values dimension should be equal to (self._horizon, self._num_models, self._num_particles*self._num_samples, 1)'
        assert costs.shape == torch.Size(
            [self._horizon, self._num_models, int(self._num_particles * self._num_action), 1],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles*self._num_samples, 1)'

        costs = costs.reshape(
            self._horizon,
            self._num_models * self._num_particles,
            self._num_action,
            1,
        )
        max_cost = torch.max(costs, dim=1).values
        sum_horizon_costs = torch.sum(max_cost, dim=0)
        mean_episode_costs = sum_horizon_costs * (1000 / self._horizon)

        rewards = rewards.reshape(
            self._horizon,
            self._num_models * self._num_particles,
            self._num_action,
            1,
        )
        values = values.reshape(
            self._horizon,
            self._num_models * self._num_particles,
            self._num_action,
            1,
        )

        sum_horizon_returns = torch.sum(rewards, dim=0) + values[-1, :, :, :]
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        mean_episode_returns = mean_particles_returns * (1000 / self._horizon)

        assert mean_episode_returns.shape[0] == self._num_action

        feasible_num = torch.sum(mean_episode_costs <= self._cost_limit).item()
        if feasible_num < self._num_elites:
            elite_values, elite_actions = -mean_episode_costs, actions
        else:
            elite_idxs = (
                (mean_episode_costs <= self._cost_limit).nonzero().reshape(-1)
            )  # like tensor([0, 1])
            elite_values, elite_actions = mean_episode_returns[elite_idxs], actions[:, elite_idxs]

        info = {
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
            'Plan/feasible_num': feasible_num,
            'Plan/episode_costs_max': mean_episode_costs.max().item(),
            'Plan/episode_costs_mean': mean_episode_costs.mean().item(),
            'Plan/episode_costs_min': mean_episode_costs.min().item(),
        }

        return elite_values, elite_actions, info
