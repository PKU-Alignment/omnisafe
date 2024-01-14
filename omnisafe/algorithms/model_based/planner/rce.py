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
"""Model Predictive Control Planner of Robust Cross Entropy algorithm."""


from __future__ import annotations

import torch

from omnisafe.algorithms.model_based.planner.cce import CCEPlanner


class RCEPlanner(CCEPlanner):
    """The planner of Robust Cross Entropy (RCE) algorithm.

    References:
        - Title: Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method
        - Authors: Zuxin Liu, Hongyi Zhou, Baiming Chen, Sicheng Zhong, Martial Hebert, Ding Zhao.
        - URL: `RCE <https://arxiv.org/abs/2010.07968>`_
    """

    @torch.no_grad()
    def _select_elites(
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
        assert costs.shape == torch.Size(
            [
                self._horizon,
                self._num_models,
                int(self._num_particles / self._num_models * self._num_samples),
                1,
            ],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles/self._num_models*self._num_samples, 1)'

        costs = costs.reshape(self._horizon, self._num_particles, self._num_samples, 1)
        max_cost = torch.max(costs, dim=1).values
        sum_horizon_costs = torch.sum(max_cost, dim=0)
        mean_episode_costs = sum_horizon_costs * (1000 / self._horizon)

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
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
            'Plan/episode_costs_max': mean_episode_costs.max().item(),
            'Plan/episode_costs_mean': mean_episode_costs.mean().item(),
            'Plan/episode_costs_min': mean_episode_costs.min().item(),
        }

        return elite_returns_topk, elite_actions_topk, info
