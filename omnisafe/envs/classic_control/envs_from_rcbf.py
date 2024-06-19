# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Interface of control barrier function-based environments."""

# mypy: ignore-errors
# pylint: disable=all

from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium import spaces


class UnicycleEnv(gymnasium.Env):
    """Environment from `The Soft Actor-Critic algorithm with Robust Control Barrier Function`."""

    def __init__(self) -> None:
        """Initialize the unicycle environment."""
        super().__init__()

        self.dynamics_mode = 'Unicycle'
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        self.bds = np.array([[-3.0, -3.0], [3.0, 3.0]])

        self.dt = 0.02
        self.max_episode_steps = 1000
        self.reward_goal = 1.0
        self.goal_size = 0.3
        self.state = None
        self.episode_step = 0
        self.initial_state = np.array(
            [[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [-2.5, 0.0, 0.0], [2.5, -2.5, np.pi / 2]],
        )
        self.goal_pos = np.array([2.5, 2.5])
        self.rand_init = False

        self.reset()

        self.get_f, self.get_g = self._get_dynamics()
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20
        self.hazards = []

        self.hazards.append(
            {'type': 'circle', 'radius': 0.6, 'location': 1.5 * np.array([0.0, 0.0])},
        )
        self.hazards.append(
            {'type': 'circle', 'radius': 0.6, 'location': 1.5 * np.array([-1.0, 1.0])},
        )
        self.hazards.append(
            {'type': 'circle', 'radius': 0.6, 'location': 1.5 * np.array([-1.0, -1.0])},
        )
        self.hazards.append(
            {'type': 'circle', 'radius': 0.6, 'location': 1.5 * np.array([1.0, -1.0])},
        )
        self.hazards.append(
            {'type': 'circle', 'radius': 0.6, 'location': 1.5 * np.array([1.0, 1.0])},
        )
        self.viewer = None

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, float, bool, bool, dict[str, Any]]:
        """Step the environment."""
        action = np.clip(action, -1.0, 1.0)
        state, reward, cost, terminated, truncated, info = self._step(action)
        return self.get_obs(), reward, cost, terminated, truncated, info

    def _step(self, action: np.ndarray) -> tuple:
        """The details of step dynamics."""
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]), 0])

        self.episode_step += 1

        dist_goal = self._goal_dist()
        reward = self.last_goal_dist - dist_goal
        self.last_goal_dist = dist_goal
        terminated = False
        if self.goal_met():
            reward += self.reward_goal
            terminated = True
        truncated = self.episode_step >= self.max_episode_steps

        cost = 0.0
        for hazard in self.hazards:
            if hazard['type'] == 'circle':
                cost += 0.1 * (
                    np.sum((self.state[:2] - hazard['location']) ** 2) < hazard['radius'] ** 2
                )

        return self.state, reward, cost, terminated, truncated, {}

    def goal_met(self) -> bool:
        """Return whether meeting the goal."""
        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset the environment."""
        self.episode_step = 0

        if self.rand_init:
            self.state = np.copy(self.initial_state[np.random.randint(self.initial_state.shape[0])])
        else:
            self.state = np.copy(self.initial_state[0])

        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), {}

    def render(self, mode: str = 'human') -> np.ndarray:
        """Get the image of the running environment."""
        raise NotImplementedError

    def get_obs(self) -> np.ndarray:
        """Given the state, this function returns corresponding observation.

        Returns:
          Observation: np.ndarray.
        """
        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()

        return np.array(
            [
                self.state[0],
                self.state[1],
                np.cos(self.state[2]),
                np.sin(self.state[2]),
                goal_compass[0],
                goal_compass[1],
                np.exp(-goal_dist),
            ],
        )

    def obs_compass(self) -> np.ndarray:
        """Return a robot-centric compass observation of a list of positions."""
        vec = self.goal_pos - self.state[:2]
        R = np.array(
            [
                [np.cos(self.state[2]), -np.sin(self.state[2])],
                [np.sin(self.state[2]), np.cos(self.state[2])],
            ],
        )
        vec = np.matmul(vec, R)
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _get_dynamics(self) -> tuple[Callable, Callable]:

        def get_f(state: np.ndarray) -> np.ndarray:
            """Function to compute the drift dynamics 'f(x)' of the system."""
            return np.zeros(state.shape)

        def get_g(state: np.ndarray) -> np.ndarray:
            """Function to compute the control dynamics 'g(x)' of the system."""
            theta = state[2]
            return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1.0]])

        return get_f, get_g

    def _goal_dist(self) -> np.ndarray:
        """Calculate the distance between the goal."""
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self) -> None:
        """Close the instance of environment."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
