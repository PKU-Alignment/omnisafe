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
"""Interface of control barrier function-based environments."""

# mypy: ignore-errors
# pylint: disable=all

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, ClassVar

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box


def to_pixel(meas_cm: list[float] | float, shift: int = 0) -> float:
    if isinstance(meas_cm, Iterable):
        return 1.5 * 37.795 * meas_cm + np.array(shift)

    return 1.5 * 37.795 * meas_cm + shift


class UnicycleEnv(gymnasium.Env):

    def __init__(self) -> None:

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
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.initial_state = np.array(
            [[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [-2.5, 0.0, 0.0], [2.5, -2.5, np.pi / 2]],
        )
        self.goal_pos = np.array([2.5, 2.5])
        self.rand_init = False

        self.reset()

        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()
        # Disturbance
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20

        # Build Hazards
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

        # Viewer
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
        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
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
        goal_compass = self.obs_compass()  # compass to the goal

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

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array(
            [
                [np.cos(self.state[2]), -np.sin(self.state[2])],
                [np.sin(self.state[2]), np.cos(self.state[2])],
            ],
        )
        vec = np.matmul(vec, R)
        # Normalize
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


@env_register
class RobustBarrierFunctionEnv(CMDP):
    """Interface of control barrier function-based environments.

    .. warning::
        Since environments based on control barrier functions require special judgment and control
        of environmental dynamics, they do not support the use of vectorized environments for
        parallelization.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    _support_envs: ClassVar[list[str]] = [
        'Unicycle',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.

        Keyword Args:
            render_mode (str, optional): The render mode, ranging from ``human``, ``rgb_array``, ``rgb_array_list``.
                Defaults to ``rgb_array``.
            camera_name (str, optional): The camera name.
            camera_id (int, optional): The camera id.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.
        """
        super().__init__(env_id)
        self._env_id = env_id
        if num_envs == 1:
            if self._env_id == 'Unicycle':
                self._env = UnicycleEnv()
            else:
                raise NotImplementedError('Only support Unicycle now.')
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._device = torch.device(device)

        self._num_envs = num_envs
        self._metadata = self._env.metadata

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        .. note::

            OmniSafe use auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode.
            And the true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            cost: Amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        """Return the unwrapped environment attributes."""
        return getattr(self._env, name)
