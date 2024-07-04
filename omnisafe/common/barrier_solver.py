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

import warnings

import numpy as np
import torch
from cvxopt import matrix, solvers

from omnisafe.typing import DEVICE_CPU


# pylint: disable-next=too-many-instance-attributes
class PendulumSolver:
    """The CBF solver for the pendulum problem using Gaussian Process models.

    This class implements a solver for the pendulum control problem using Control Barrier Functions
    (CBFs). The primary goal is to ensure safe reinforcement learning by maintaining
    safety constraints during the control process.

    For more details, please refer to:

    *End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous
    Control Tasks*

    Attributes:
        action_size (int): Size of the action space, typically 1 for the pendulum.
        torque_bound (float): Maximum torque bound that can be applied to the pendulum.
        max_speed (float): Maximum speed (angular velocity) of the pendulum.
        device (torch.device): Device to run the computations on.
    """

    # pylint: disable-next=invalid-name
    def __init__(
        self,
        action_size: int = 1,
        torque_bound: float = 15.0,
        max_speed: float = 60.0,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize the PendulumSolver with specified parameters.

        Args:
            action_size (int): Size of the action space, typically 1 for the pendulum.
            torque_bound (float): Maximum torque bound that can be applied to the pendulum.
            max_speed (float): Maximum speed (angular velocity) of the pendulum.
            device (torch.device): Device to run the computations on.

        Attributes:
            F (float): A control gain factor used in the CBF computation.
            _gamma_b (float): Parameter for the barrier function.
            _kd (float): Damping coefficient used in the barrier function.
        """
        self.action_size = action_size
        self.torque_bound = torque_bound
        self.max_speed = max_speed
        self.F = 1.0
        self._device = device
        self._gamma_b = 0.5
        self._kd = 1.5
        self._build_barrier()
        warnings.filterwarnings('ignore')

    def _build_barrier(self) -> None:
        """Construct the Control Barrier Function (CBF) for safe control of the pendulum.

        This method initializes and sets up the necessary components for the CBF, which is used to
        ensure that the control actions taken do not violate safety constraints.
        """
        self.P = matrix(np.diag([1.0, 1e16]), tc='d')
        self.q = matrix(np.zeros(self.action_size + 1))
        self.h1 = np.array([1, 0.01])
        self.h2 = np.array([1, -0.01])
        self.h3 = np.array([-1, 0.01])
        self.h4 = np.array([-1, -0.01])

    def control_barrier(  # pylint: disable=invalid-name
        self,
        original_action: torch.Tensor,
        f: np.ndarray,
        g: np.ndarray,
        x: np.ndarray,
        std: np.ndarray,
    ) -> torch.Tensor:
        """Adjust the original action using a control barrier function.

        Args:
            original_action (torch.Tensor): The original action proposed by the RL algorithm.
            f (np.ndarray): The drift component of the system's dynamics.
            g (np.ndarray): The control component of the system's dynamics.
            x (np.ndarray): The current state of the system.
            std (np.ndarray): The standard deviation of the system's state.

        Returns:
            torch.Tensor: The adjusted action that respects the system's constraints.
        """
        # define gamma for the barrier function
        gamma_b = 0.5
        kd = 1.5
        u_rl = original_action.cpu().detach().numpy()

        # set up Quadratic Program to satisfy Control Barrier Function
        G = np.array(
            [
                [
                    -np.dot(self.h1, g),
                    -np.dot(self.h2, g),
                    -np.dot(self.h3, g),
                    -np.dot(self.h4, g),
                    1,
                    -1,
                    g[1],
                    -g[1],
                ],
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    0,
                    0,
                    0,
                ],
            ],
        )
        G = np.transpose(G)
        h = np.array(
            [
                gamma_b * self.F
                + np.dot(self.h1, f)
                + np.dot(self.h1, g) * u_rl
                - (1 - gamma_b) * np.dot(self.h1, x)
                - kd * np.abs(np.dot(self.h1, std)),
                gamma_b * self.F
                + np.dot(self.h2, f)
                + np.dot(self.h2, g) * u_rl
                - (1 - gamma_b) * np.dot(self.h2, x)
                - kd * np.abs(np.dot(self.h2, std)),
                gamma_b * self.F
                + np.dot(self.h3, f)
                + np.dot(self.h3, g) * u_rl
                - (1 - gamma_b) * np.dot(self.h3, x)
                - kd * np.abs(np.dot(self.h3, std)),
                gamma_b * self.F
                + np.dot(self.h4, f)
                + np.dot(self.h4, g) * u_rl
                - (1 - gamma_b) * np.dot(self.h4, x)
                - kd * np.abs(np.dot(self.h4, std)),
                -u_rl + self.torque_bound,
                u_rl + self.torque_bound,
                -f[1] - g[1] * u_rl + self.max_speed,
                f[1] + g[1] * u_rl + self.max_speed,
            ],
        )
        h = np.squeeze(h).astype(np.double)

        # convert numpy arrays to cvx matrices to set up QP
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        solvers.options['show_progress'] = False
        sol = solvers.qp(self.P, self.q, G, h)
        u_bar = sol['x']

        # check if the adjusted action is within bounds
        if np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= self.torque_bound:
            u_bar[0] = self.torque_bound - u_rl
            print('Error in QP')
        elif np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= -self.torque_bound:
            u_bar[0] = -self.torque_bound - u_rl
            print('Error in QP')

        return torch.as_tensor(u_bar[0], dtype=torch.float32, device=self._device).unsqueeze(dim=0)
