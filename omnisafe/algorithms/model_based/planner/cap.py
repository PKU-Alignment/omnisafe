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
"""Model Predictive Control Planner of the Conservative and Adaptive Penalty algorithm."""


from __future__ import annotations

from typing import Any

import torch

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.cce import CCEPlanner
from omnisafe.utils.config import Config


class CAPPlanner(CCEPlanner):
    """The planner of Conservative and Adaptive Penalty (CAP) algorithm.

    References:
        - Title: Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning
        - Authors: Yecheng Jason Ma, Andrew Shen, Osbert Bastani, Dinesh Jayaraman.
        - URL: `CAP <https://arxiv.org/abs/2112.07701>`_
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self,
        dynamics: EnsembleDynamicsModel,
        planner_cfgs: Config,
        gamma: float,
        cost_gamma: float,
        dynamics_state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        action_max: float,
        action_min: float,
        device: torch.device,
        **kwargs: Any,
    ) -> None:
        """Initializes the planner of Conservative and Adaptive Penalty (CAP) algorithm."""
        super().__init__(
            dynamics,
            planner_cfgs,
            gamma,
            cost_gamma,
            dynamics_state_shape,
            action_shape,
            action_max,
            action_min,
            device,
            **kwargs,
        )
        self._lagrange: torch.Tensor = kwargs['lagrange']

    @torch.no_grad()
    def _select_elites(  # pylint: disable=too-many-locals
        self,
        actions: torch.Tensor,
        traj: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Select elites from the sampled actions.

        Args:
            actions (torch.Tensor): Sampled actions.
            traj (dict[str, torch.Tensor]): Trajectory dictionary.

        Returns:
            elites_value: The value of the elites.
            elites_action: The action of the elites.
            info: The dictionary containing the information of elites value and action.
        """
        rewards = traj['rewards']
        costs = traj['costs']
        state_vars = traj['vars']
        assert actions.shape == torch.Size(
            [self._horizon, self._num_samples, *self._action_shape],
            # pylint: disable-next=line-too-long
        ), 'Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)'
        assert rewards.shape == torch.Size(
            [
                self._horizon,
                self._num_models,
                int(self._num_particles / self._num_models * self._num_samples),
                1,
            ],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)'
        assert state_vars.shape[:-1] == torch.Size(
            [
                self._horizon,
                self._num_models,
                int(self._num_particles / self._num_models * self._num_samples),
            ],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, dynamics_state_shape)'
        assert costs.shape == torch.Size(
            [
                self._horizon,
                self._num_models,
                int(self._num_particles / self._num_models * self._num_samples),
                1,
            ],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)'
        # var: [horizon, network_size, num_gaussian_traj*particles/network_size, state_dim]
        var_penalty = state_vars.sqrt().norm(dim=3).max(1)[0]
        # cost_penalty: [horizon, num_gaussian_traj*particles/network_size]
        var_penalty = var_penalty.repeat_interleave(self._num_models).view(
            costs.shape,
        )
        # cost_penalty: [horizon, num_gaussian_traj*particle]
        costs += self._lagrange * var_penalty

        costs = costs.reshape(self._horizon, self._num_particles, self._num_samples, 1)
        sum_horizon_costs = torch.sum(costs, dim=0)
        mean_particles_costs = sum_horizon_costs.mean(dim=0)
        mean_episode_costs = mean_particles_costs * (1000 / self._horizon)

        returns = rewards.reshape(self._horizon, self._num_particles, self._num_samples, 1)
        sum_horizon_returns = torch.sum(returns, dim=0)
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        mean_episode_returns = mean_particles_returns * (1000 / self._horizon)

        assert mean_particles_returns.shape[0] == self._num_samples

        feasible_num = torch.sum(mean_episode_costs <= self._cost_limit).item()
        if feasible_num < self._num_elites:
            elite_values, elite_actions = -mean_episode_costs, actions
        else:
            elite_idxs = (
                (mean_episode_costs <= self._cost_limit).nonzero().reshape(-1)
            )  # like tensor([0, 1])
            elite_values, elite_actions = mean_episode_returns[elite_idxs], actions[:, elite_idxs]

        elite_idxs_topk = torch.topk(elite_values.squeeze(1), self._num_elites, dim=0).indices
        elite_returns_topk, elite_actions_topk = (
            elite_values[elite_idxs_topk],
            elite_actions[:, elite_idxs_topk],
        )
        info = {
            'Plan/feasible_num': feasible_num,
            'Plan/var_penalty_max': var_penalty.max().item(),
            'Plan/var_penalty_mean': var_penalty.mean().item(),
            'Plan/var_penalty_min': var_penalty.min().item(),
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
            'Plan/episode_costs_max': mean_episode_costs.max().item(),
            'Plan/episode_costs_mean': mean_episode_costs.mean().item(),
            'Plan/episode_costs_min': mean_episode_costs.min().item(),
        }

        return elite_returns_topk, elite_actions_topk, info
