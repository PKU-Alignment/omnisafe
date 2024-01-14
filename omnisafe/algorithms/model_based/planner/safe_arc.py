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
"""Model Predictive Control Planner of the Safe Actor Regularized Control algorithm."""


from __future__ import annotations

from typing import Any

import torch

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.arc import ARCPlanner
from omnisafe.utils.config import Config


class SafeARCPlanner(ARCPlanner):
    """The planner of Safe Actor Regularized Control (ARC) algorithm.

    References:
        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `Safe ARC <https://arxiv.org/abs/2008.10066>`_
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
        """Initializes the planner of Safe Actor Regularized Control (ARC) algorithm."""
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
        self._cost_limit: float = kwargs['cost_limit']
        self._cost_temperature: float = planner_cfgs.cost_temperature

    @torch.no_grad()
    def _update_mean_var(
        self,
        elite_actions: torch.Tensor,
        elite_values: torch.Tensor,
        info: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the mean and variance of the elite actions.

        Args:
            elite_actions (torch.Tensor): The elite actions.
            elite_values (torch.Tensor): The elite values.
            info (dict[str, float]): The dictionary containing the information of the elite values and actions.

        Returns:
            new_mean: The new mean of the elite actions.
            new_var: The new variance of the elite actions.
        """
        assert (
            elite_actions.shape[0] == self._horizon
            and elite_actions.shape[-1] == self._action_shape[0]
        ), 'Input elite_actions dimension should be equal to (self._horizon, self._num_elites, self._action_shape)'
        assert (
            elite_values.shape[-1] == 1
        ), 'Input elite_values dimension should be equal to (self._num_elites, 1)'
        assert (
            elite_actions.shape[1] == elite_values.shape[0]
        ), 'Number of action should be the same'
        use_cost_temperature = info['Plan/feasible_num'] < self._num_elites

        max_value = elite_values.max(0)[0]
        if use_cost_temperature is True:
            score = torch.exp(self._cost_temperature * (elite_values - max_value))
        else:
            score = torch.exp(self._temperature * (elite_values - max_value))
        score /= score.sum(0)
        new_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
        new_var = torch.sum(
            score.unsqueeze(0) * (elite_actions - new_mean.unsqueeze(1)) ** 2,
            dim=1,
        ) / (score.sum(0) + 1e-9)
        new_var = new_var.clamp_(0, 2)

        return new_mean, new_var

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

    @torch.no_grad()
    def output_action(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Output the action given the state.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            action: The action of the agent.
            info: The dictionary containing the information of the action.
        """
        assert state.shape == torch.Size(
            [1, *self._dynamics_state_shape],
        ), 'Input state dimension should be equal to (1, self._dynamics_state_shape)'
        last_mean = torch.zeros_like(self._action_sequence_mean)
        last_var = self._action_sequence_var.clone()
        last_mean[:-1] = self._action_sequence_mean[1:].clone()
        last_mean[-1] = self._action_sequence_mean[-1].clone()

        current_iter = 0
        actions_actor = self._act_from_actor(state)
        info: dict[str, float] = {}
        while current_iter < self._num_iterations and last_var.max() > self._epsilon:
            actions_gauss = self._act_from_last_gaus(last_mean=last_mean, last_var=last_var)
            actions = torch.cat([actions_gauss, actions_actor], dim=1)
            # [horizon, num_sample, action_shape]
            states_repeat, actions_repeat = self._state_action_repeat(state, actions)
            # pylint: disable-next=line-too-long
            # [num_particles * num_samples/num_ensemble, state_shape], [horizon, num_particles * num_samples/num_ensemble, action_shape]
            traj = self._dynamics.imagine(states_repeat, self._horizon, actions_repeat)
            # pylint: disable-next=line-too-long
            # {states, rewards, values}, each value shape is [horizon, num_ensemble, num_particles * num_samples/num_ensemble, 1]

            elite_values, elite_actions, info = self._select_elites(actions, traj)
            # [num_sample, 1]
            new_mean, new_var = self._update_mean_var(
                elite_actions,
                elite_values,
                info,
            )
            # last_mean = self._momentum * last_mean + (1 - self._momentum) * new_mean
            last_mean = new_mean
            last_var = self._momentum * last_var + (1 - self._momentum) * new_var
            current_iter += 1
        logger_info = {
            'Plan/iter': current_iter,
            'Plan/last_var_mean': last_var.mean().item(),
            'Plan/last_var_max': last_var.max().item(),
            'Plan/last_var_min': last_var.min().item(),
        }
        logger_info.update(info)
        self._action_sequence_mean = last_mean.clone()
        return last_mean[0].clone().unsqueeze(0), logger_info
