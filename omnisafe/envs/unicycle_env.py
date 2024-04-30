from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def to_pixel(meas_cm: list[float] | float, shift: int = 0) -> float:
    """Convert measurements from centimeters to pixels.

    Args:
        meas_cm (list[float] | float): A single measurement or a list of measurements in centimeters.
        shift (int, optional): An integer value that is added to the converted measurement(s). Default is 0.

    Returns:
        float | np.ndarray: The measurement converted to pixels.
    """
    if isinstance(meas_cm, Iterable):
        return 1.5 * 37.795 * meas_cm + np.array(shift)

    return 1.5 * 37.795 * meas_cm + shift


class UnicycleEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

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
        """
        Advance the environment state based on the action taken by the agent.

        Parameters:
            action(np.ndarray): Control action taken by the agent.

        Returns:
            A tuple containing:
            - new_obs : np.ndarray, the new observation structured as [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal].
            - reward : float, reward received after taking the action.
            - cost : float, cost incurred after taking the action.
            - terminated : bool, whether the episode has terminated.
            - truncated : bool, whether the episode was truncated.
            - info : dict, additional information about the environment's state.
        """
        action = np.clip(action, -1.0, 1.0)
        state, reward, cost, terminated, truncated, info = self._step(action)
        return self.get_obs(), reward, cost, terminated, truncated, info

    def _step(self, action: np.ndarray) -> tuple:
        """
        Update the internal state based on the action, considering dynamics and disturbances.

        Parameters:
            action(np.ndarray): Control action taken by the agent.

        Returns:
            A tuple containing:
            - state : np.ndarray, new internal state of the agent.
            - reward : float, reward collected during this transition.
            - cost : float, cost incurred during this transition.
            - terminated : bool, whether the episode has terminated.
            - truncated : bool, whether the episode was truncated due to reaching a step limit.
            - info : dict, additional information relevant to the environment.
        """
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
        """
        Check if the current goal has been met in this step.

        Returns:
            True if the agent has reached the goal, False otherwise.
        """
        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Reset the environment to an initial state.

        Parameters:
            seed : int, optional
                Seed for random number generator.
            options : dict, optional
                Additional options to customize the environment reset.

        Returns:
            A tuple containing:
            - observation : np.ndarray, the first observation after reset.
            - info : dict, additional information about the reset state.
        """
        self.episode_step = 0

        if self.rand_init:
            self.state = np.copy(self.initial_state[np.random.randint(self.initial_state.shape[0])])
        else:
            self.state = np.copy(self.initial_state[0])

        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), {}

    def render(self, mode: str = 'human') -> np.ndarray:
        """Render the environment to the screen

        Parameters:---
        mode : str
        close : bool

        Returns:

        """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print(
                f'Ep_step = {self.episode_step}, \tState = {self.state}, \tDist2Goal = {self._goal_dist()}, alignment_error = {theta_error}',
            )

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from envs import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards)):
                if self.hazards[i]['type'] == 'circle':
                    obstacles.append(
                        pyglet_rendering.make_circle(
                            radius=to_pixel(self.hazards[i]['radius'], shift=0),
                            filled=True,
                        ),
                    )
                    obs_trans = pyglet_rendering.Transform(
                        translation=(
                            to_pixel(self.hazards[i]['location'][0], shift=screen_width / 2),
                            to_pixel(self.hazards[i]['location'][1], shift=screen_height / 2),
                        ),
                    )
                    obstacles[i].set_color(1.0, 0.0, 0.0)
                    obstacles[i].add_attr(obs_trans)
                elif self.hazards[i]['type'] == 'polygon':
                    obstacles.append(
                        pyglet_rendering.make_polygon(
                            to_pixel(
                                self.hazards[i]['vertices'],
                                shift=[screen_width / 2, screen_height / 2],
                            ),
                            filled=True,
                        ),
                    )
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(
                translation=(
                    to_pixel(self.goal_pos[0], shift=screen_width / 2),
                    to_pixel(self.goal_pos[1], shift=screen_height / 2),
                ),
            )
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(
                translation=(
                    to_pixel(self.state[0], shift=screen_width / 2),
                    to_pixel(self.state[1], shift=screen_height / 2),
                ),
            )
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

        if self.state is None:
            return None

        self.robot_trans.set_translation(
            to_pixel(self.state[0], shift=screen_width / 2),
            to_pixel(self.state[1], shift=screen_height / 2),
        )
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_obs(self) -> np.ndarray:
        """Given the state, this function returns corresponding observation.

        Returns:
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
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

    def _get_dynamics(self) -> tuple[Callable, Callable]:
        """Get affine Control Barrier Function (CBF) dynamics for a given environment.

        This method provides access to the system's drift and control dynamics, formulated for continuous systems of the form x' = f(x) + g(x)u, where 'x' is the state vector and 'u' is the control vector.

        Returns:
            get_f : Callable[[np.ndarray], np.ndarray]
                Function to compute the drift dynamics 'f(x)' of the system.

            get_g : Callable[[np.ndarray], np.ndarray]
                Function to compute the control dynamics 'g(x)' of the system.
        """

        def get_f(state: np.ndarray) -> np.ndarray:
            """Function to compute the drift dynamics 'f(x)' of the system."""
            return np.zeros(state.shape)

        def get_g(state: np.ndarray) -> np.ndarray:
            """Function to compute the control dynamics 'g(x)' of the system."""
            theta = state[2]
            return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1.0]])

        return get_f, get_g

    def obs_compass(self) -> np.ndarray:
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

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

    def _goal_dist(self) -> np.ndarray:
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_random_hazard_locations(self, n_hazards: int, hazard_radius: float) -> None:
        """

        Parameters:---
        n_hazards : int
            Number of hazards to create
        hazard_radius : float
            Radius of hazards

        Returns:
        hazards_locs : np.ndarray
            Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
        """

        # Create buffer with boundaries
        buffered_bds = np.copy(self.bds)
        buffered_bds[0] = buffered_bds[0] + hazard_radius
        buffered_bds[1] -= hazard_radius

        hazards = []
        hazards_centers = np.zeros((n_hazards, 2))
        n = 0  # Number of hazards actually placed
        for _ in range(n_hazards):
            successfully_placed = False
            iteration = 0
            hazard_type = np.random.randint(3)  # 0-> Circle 1->Square 2->Triangle
            radius = hazard_radius * (1 - 0.2 * 2.0 * (np.random.random() - 0.5))
            while not successfully_placed and iteration < 100:
                hazards_centers[n] = (buffered_bds[1] - buffered_bds[0]) * np.random.random(
                    2,
                ) + buffered_bds[0]
                successfully_placed = np.all(
                    np.linalg.norm(hazards_centers[:n] - hazards_centers[[n]], axis=1)
                    > 3.5 * hazard_radius,
                )
                successfully_placed = np.logical_and(
                    successfully_placed,
                    np.linalg.norm(self.goal_pos - hazards_centers[n]) > 2.0 * hazard_radius,
                )
                successfully_placed = np.logical_and(
                    successfully_placed,
                    np.all(
                        np.linalg.norm(self.initial_state[:, :2] - hazards_centers[[n]], axis=1)
                        > 2.0 * hazard_radius,
                    ),
                )
                iteration += 1
            if not successfully_placed:
                continue
            if hazard_type == 0:  # Circle
                hazards.append({'type': 'circle', 'location': hazards_centers[n], 'radius': radius})
            elif hazard_type == 1:  # Square
                hazards.append(
                    {
                        'type': 'polygon',
                        'vertices': np.array(
                            [
                                [-radius, -radius],
                                [-radius, radius],
                                [radius, radius],
                                [radius, -radius],
                            ],
                        ),
                    },
                )
                hazards[-1]['vertices'] += hazards_centers[n]
            else:  # Triangle
                hazards.append(
                    {
                        'type': 'polygon',
                        'vertices': np.array(
                            [
                                [-radius, -radius],
                                [-radius, radius],
                                [radius, radius],
                                [radius, -radius],
                            ],
                        ),
                    },
                )
                # Pick a vertex and delete it
                idx = np.random.randint(4)
                hazards[-1]['vertices'] = np.delete(hazards[-1]['vertices'], idx, axis=0)
                hazards[-1]['vertices'] += hazards_centers[n]
            n += 1

        self.hazards = hazards
