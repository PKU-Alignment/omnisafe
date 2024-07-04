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
"""Implementation of the Control Barrier Function Solver."""

# pylint: disable=invalid-name,wrong-spelling-in-docstring
# mypy: ignore-errors


from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C


# pylint: disable-next=too-many-instance-attributes
class DynamicsModel:
    """This class handles the creation and management of Gaussian Process (GP) models.

    These GP models predict the next state of the environment based on the current state.

    .. warning::
        This class provides an implementation for the  ``Pendulum-v1``  environment. It needs to be
        customized to extend it to more environments.

    Args:
        observation_size (int): The size of the observation space. This determines
                                the number of GP models to create.
        load_dir (Optional[str]): The directory to load the GP models from. If None, new models
                                  are initialized. Default is None.

    Attributes:
        observation_size (int): The size of the observation space.
        gp_model_prev (List[GaussianProcessRegressor]): The GP models from the previous iteration.
        gp_model (List[GaussianProcessRegressor]): The current GP models used for predictions.
    """

    def __init__(self, observation_size: int, load_dir: str | None = None) -> None:
        """Initialize the DynamicsModel with a specified observation size and optional model loading.

        Args:
            observation_size (int): Size of the observation space.
            load_dir (Optional[str]): Directory to load the GP models from. If not provided,
                                      new models will be created.
        """
        self.observation_size: int = observation_size
        self.gp_model_prev: list[GaussianProcessRegressor]
        self.gp_model: list[GaussianProcessRegressor]
        self._build_gp_model(load_dir=load_dir)

    def _build_gp_model(self, load_dir: str | None = None) -> None:
        """Build or load the Gaussian Process models.

        If a load directory is provided, the models are loaded from the specified directory.
        Otherwise, new models are created with default parameters.

        Args:
            load_dir (Optional[str]): Directory to load the GP models from. If None, new models
                                      will be created.
        """
        gp_list = []
        noise = 0.01  # Small noise term to stabilize the GP model
        for _ in range(self.observation_size - 1):
            if not load_dir:
                # Define the kernel as a product of a constant kernel and an RBF kernel
                kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                # Initialize the GaussianProcessRegressor with the specified kernel and noise
                gp = GaussianProcessRegressor(kernel=kern, alpha=noise, n_restarts_optimizer=10)
                gp_list.append(gp)
            else:
                # Load the GP models from the specified directory
                gp_list = joblib.load(load_dir)
        self.gp_model = gp_list
        self.gp_model_prev = gp_list.copy()

    @property
    def gp_models(self) -> list[GaussianProcessRegressor]:
        """Return all gaussian process regressor for saving."""
        return self.gp_model

    def get_dynamics(self, obs: list[float], original_action: float) -> np.ndarray:
        """Calculate the dynamics of the system based on the current observation and the original action.

        This method computes the next state of a pendulum system using the provided state and
        action.

        Args:
            obs (list[float]): The current observation of the system state.
                               For the ``Pendulum-v1``, It should contain at least three elements:
                               [x, y, theta_dot], where x and y are the Cartesian coordinates of
                               the pendulum, and theta_dot is the angular velocity.
            original_action (float): The original action proposed by the RL agent.

        Returns:
            np.ndarray: The calculated dynamics of the system, representing the next state.
        """
        # Time step
        dt = 0.05
        # Gravitational constant
        G = 10
        # Mass of the pendulum
        m = 2
        # Length of the pendulum
        length = 2

        # Calculate the angle theta from the Cartesian coordinates
        theta = np.arctan2(obs[1], obs[0])
        # Angular velocity
        theta_dot = obs[2]

        f = np.array(
            [
                -3 * G / (2 * length) * np.sin(theta + np.pi) * dt**2
                + theta_dot * dt
                + theta
                + 3 / (m * length**2) * original_action * dt**2,
                theta_dot
                - 3 * G / (2 * length) * np.sin(theta + np.pi) * dt
                + 3 / (m * length**2) * original_action * dt,
            ],
        )

        return np.squeeze(f)

    def update_gp_dynamics(self, obs: np.ndarray, act: np.ndarray) -> None:
        """Update the Gaussian Process (GP) dynamics model based on observed states and actions.

        Args:
            obs (np.ndarray): Agent's observation of the current environment.
            act (np.ndarray): Actions taken.
        """
        obs = obs.detach().cpu().squeeze().numpy()
        act = act.detach().cpu().squeeze().numpy()
        N = self.observation_size
        X = obs
        U = act
        L = len(X)
        err = np.zeros((L - 1, N - 1))
        S = np.zeros((L - 1, 2))
        for i in range(L - 1):
            f = self.get_dynamics(X[i], U[i])
            theta_p = np.arctan2(X[i][1], X[i][0])
            theta_dot_p = X[i][2]
            theta = np.arctan2(X[i + 1][1], X[i + 1][0])
            theta_dot = X[i + 1][2]
            S[i, :] = np.array([theta_p, theta_dot_p])
            err[i, :] = np.array([theta, theta_dot]) - f
        self.gp_model[0].fit(S, err[:, 0])
        self.gp_model[1].fit(S, err[:, 1])

    def get_gp_dynamics(self, obs: torch.Tensor, use_prev_model: bool) -> list[np.ndarray]:
        """Retrieve the GP dynamics based on the current observation.

        Args:
            obs (torch.Tensor): Agent's observation of the current environment.
            use_prev_model (bool): Whether to use previous gaussian model.

        Returns:
            list[np.ndarray]: list containing the gp dynamics [f, g, x, std].
        """
        obs = obs.cpu().detach().numpy()
        u_rl = 0
        dt = 0.05
        G = 10
        m = 1
        length = 1
        obs = np.squeeze(obs)
        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]
        x = np.array([theta, theta_dot])
        f_nom = np.array(
            [
                -3 * G / (2 * length) * np.sin(theta + np.pi) * dt**2
                + theta_dot * dt
                + theta
                + 3 / (m * length**2) * u_rl * dt**2,
                theta_dot
                - 3 * G / (2 * length) * np.sin(theta + np.pi) * dt
                + 3 / (m * length**2) * u_rl * dt,
            ],
        )
        g = np.array([3 / (m * length**2) * dt**2, 3 / (m * length**2) * dt])
        f_nom = np.squeeze(f_nom)
        f = np.zeros(2)
        if use_prev_model:
            [m1, std1] = self.gp_model_prev[0].predict(x.reshape(1, -1), return_std=True)
            [m2, std2] = self.gp_model_prev[1].predict(x.reshape(1, -1), return_std=True)
        else:
            [m1, std1] = self.gp_model[0].predict(x.reshape(1, -1), return_std=True)
            [m2, std2] = self.gp_model[1].predict(x.reshape(1, -1), return_std=True)
        f[0] = f_nom[0] + m1
        f[1] = f_nom[1] + m2
        return [
            np.squeeze(f),
            np.squeeze(g),
            np.squeeze(x),
            np.array([np.squeeze(std1), np.squeeze(std2)]),
        ]

    def reset_gp_model(self) -> None:
        """Reset the gaussian process model of barrier function solver."""
        self.gp_model_prev = self.gp_model.copy()
        self._build_gp_model()
