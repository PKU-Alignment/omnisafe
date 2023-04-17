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
"""Model Predictive Control Planner of the Actor Regularized Control (ARC) algorithm."""
from __future__ import annotations

import torch


class ARCPlanner:  # pylint: disable=too-many-instance-attributes
    """The planner of Actor Regularized Control (ARC) algorithm.

    References:

        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `ARC <https://arxiv.org/abs/2008.10066>`_
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
        init_var,
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
        self._action_sequence_mean = torch.zeros(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )
        self._init_var = init_var
        self._action_sequence_var = init_var * torch.ones(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )

    @torch.no_grad()
    def _act_from_last_gaus(self, last_mean: torch.Tensor, last_var: torch.Tensor) -> torch.Tensor:
        """Sample actions from the last gaussian distribution.

        Args:
            last_mean (torch.Tensor): Last mean of the gaussian distribution.
            last_var (torch.Tensor): Last variance of the gaussian distribution.

        Returns:
            Sample actions (torch.Tensor): Sampled actions from the last gaussian distribution.
        """
        constrained_std = torch.sqrt(last_var)
        actions = torch.clamp(
            last_mean.unsqueeze(1)
            + constrained_std.unsqueeze(1)
            * torch.randn(
                self._horizon,
                self._num_samples,
                *self._action_shape,
                device=self._device,
            ),
            self._action_min,
            self._action_max,
        )
        actions.clamp_(min=self._action_min, max=self._action_max)  # Clip action range
        return actions

    @torch.no_grad()
    def _act_from_actor(self, state: torch.Tensor) -> torch.Tensor:
        """Sample actions from the actor.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            Sample actions (torch.Tensor): Sampled actions from the actor.
        """
        assert state.shape == torch.Size(
            [1, *self._dynamics_state_shape],
        ), 'state dimension one should be 1'
        assert (
            self._actor_traj % self._num_models == 0
        ), 'actor_traj should be divisible by num_models'
        traj = self._dynamics.imagine(
            states=state,
            horizon=self._horizon,
            actions=None,
            actor_critic=self._actor_critic,
            idx=0,
        )
        return (
            traj['actions']
            .reshape(self._horizon, 1, *self._action_shape)
            .clone()
            .repeat([1, self._actor_traj, 1])
        )

    @torch.no_grad()
    def _state_action_repeat(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Repeat the state for num_repeat * action.shape[0] times and action for num_repeat times.

        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Sampled actions.

        Returns:
            states (torch.Tensor): Repeated states.
            actions (torch.Tensor): Repeated actions.
        """
        assert action.shape == torch.Size(
            [self._horizon, self._num_action, *self._action_shape],
        ), 'Input action dimension should be equal to (self._num_samples, self._action_shape)'
        assert state.shape == torch.Size(
            [1, *self._dynamics_state_shape],
        ), 'state dimension one should be 1'
        states = state.repeat(int(self._num_particles * self._num_action), 1)
        actions = action.unsqueeze(1).repeat(1, int(self._num_particles), 1, 1)
        actions = actions.reshape(
            self._horizon,
            int(self._num_particles * self._num_action),
            *self._action_shape,
        )
        return states, actions

    @torch.no_grad()
    def _select_elites(
        self,
        actions: torch.Tensor,
        traj: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Select elites from the sampled actions.

        Args:
            actions (torch.Tensor): Sampled actions.
            traj (dict): Trajectory dictionary.

        Returns:
            elites_value (torch.Tensor): Value of the elites.
            elites_action (torch.Tensor): Action of the elites.
            info (dict): Dictionary containing the information of elites value and action.
        """
        rewards = traj['rewards']
        values = traj['values']
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

        elite_actions = actions
        elite_values = mean_episode_returns
        info = {
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
        }

        return elite_values, elite_actions, info

    @torch.no_grad()
    def _update_mean_var(  # pylint: disable=unused-argument
        self,
        elite_actions: torch.Tensor,
        elite_values: torch.Tensor,
        use_cost_temperature: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the mean and variance of the elite actions.

        Args:
            elite_actions (torch.Tensor): Elite actions.
            elite_values (torch.Tensor): Elite values.
            kwargs (dict): Keyword arguments.

        Returns:
            new_mean (torch.Tensor): New mean of the elite actions.
            new_var (torch.Tensor): New variance of the elite actions.
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

        max_value = elite_values.max(0)[0]
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
    def output_action(self, state: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Output the action given the state.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            action (torch.Tensor): Action of the environment.
            logger_info (dict): Dictionary containing the information of the action.
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
                use_cost_temperature=False,
            )
            last_mean = self._momentum * last_mean + (1 - self._momentum) * new_mean
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

    def reset_planner(self):
        """Reset the planner."""
        self._action_sequence_mean = torch.zeros(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )
        self._action_sequence_var = self._init_var * torch.ones(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )
