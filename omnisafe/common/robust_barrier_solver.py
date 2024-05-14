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
"""Robust Control Barrier Function Solver for OmniSafe."""


# mypy: ignore-errors
# pylint: disable=invalid-name,wrong-spelling-in-docstring
from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import torch
from qpth.qp import QPFunction

from omnisafe.utils.tools import to_tensor


DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2}}


class CBFQPLayer:
    """CBFQLayer for robust control barrier function solver.

    Args:
        env (gymnasium.Env): The Gymnasium environment to interact with.
        device (str, optional): The device type, such as 'cpu' or 'gpu'. Defaults to 'cpu'.
        gamma_b (float, optional): The gamma parameter. Defaults to 20.
        l_p (float, optional): Some additional layer parameter, purpose unspecified. Defaults to 0.03.

    Attributes:
        device (torch.device): The device on which computations will be performed.
        env (gym.Env): The Gym environment instance.
        u_min (float): The minimum control bound.
        u_max (float): The maximum control bound.
        gamma_b (float): The gamma parameter.
        l_p (float): An additional layer parameter.
        action_dim (int): The dimensionality of the action space.
    """

    def __init__(
        self,
        env: gym.Env,
        device: str = 'cpu',
        gamma_b: float = 20,
        l_p: float = 0.03,
    ) -> None:
        """Initializes a CBFLayer instance with specified parameters and environment."""
        self.device = torch.device(device)
        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b
        self.l_p = l_p
        self.action_dim = env.action_space.shape[0]
        warnings.filterwarnings('ignore')

    def get_safe_action(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        mean_pred_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Computes safe actions based on current state and action predictions, adjusting for uncertainties.

        Args:
            state_batch (torch.Tensor): Current state batch, tensor or ndarray.
            action_batch (torch.Tensor): Nominal action batch, tensor or ndarray.
            mean_pred_batch (torch.Tensor): Mean disturbance predictions, tensor or ndarray.
            sigma_batch (torch.Tensor): Standard deviations of disturbances, tensor or ndarray.

        Returns:
            torch.Tensor: Safe actions adjusted for given constraints and uncertainties.
        """
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = state_batch.unsqueeze(0)
            action_batch = action_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        Ps, qs, Gs, hs = self.get_cbf_qp_constraints(
            state_batch,
            action_batch,
            mean_pred_batch,
            sigma_batch,
        )
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        final_action_batch = torch.clamp(
            action_batch + safe_action_batch,
            self.u_min.repeat(action_batch.shape[0], 1),
            self.u_max.repeat(action_batch.shape[0], 1),
        )

        return final_action_batch if not expand_dims else final_action_batch.squeeze(0)

    def solve_qp(
        self,
        Ps: torch.Tensor,
        qs: torch.Tensor,
        Gs: torch.Tensor,
        hs: torch.Tensor,
    ) -> torch.Tensor:
        """Solves a batch of quadratic programming (QP) problems.

        Each QP problem is defined as:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
            subject to G[u,eps]^T <= h

        Args:
            Ps (torch.Tensor): Quadratic cost matrix for each problem.
            qs (torch.Tensor): Linear cost vector for each problem.
            Gs (torch.Tensor): Inequality constraint matrix for each problem.
            hs (torch.Tensor): Inequality constraint vector for each problem.

        Returns:
            The safe action for each problem, omitting the slack variable, with dimension (batch_size, n_u).
        """
        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        sol = self.cbf_layer(
            Ps,
            qs,
            Gs,
            hs,
            solver_args={
                'check_Q_spd': False,
                'maxIter': 100000,
                'notImprovedLim': 10,
                'eps': 1e-4,
            },
        )

        return sol[:, : self.env.action_space.shape[0]]

    def cbf_layer(
        self,
        Qs: torch.Tensor,
        ps: torch.Tensor,
        Gs: torch.Tensor,
        hs: torch.Tensor,
        As: torch.Tensor | None = None,
        bs: torch.Tensor | None = None,
        solver_args: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Applies a custom layer to solve QP problems using given constraints.

        Args:
            Qs (torch.Tensor): Quadratic cost matrix for each problem.
            ps (torch.Tensor): Linear cost vector for each problem.
            Gs (torch.Tensor): Inequality constraint matrix for each problem.
            hs (torch.Tensor): Inequality constraint vector for each problem.
            As (torch.Tensor, optional): Equality constraint matrix. Defaults to None.
            bs (torch.Tensor, optional): Equality constraint vector. Defaults to None.
            solver_args (dict, optional): Dictionary of solver arguments. Defaults to None.

        Returns:
            Result of the QP solver for each problem.
        """
        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        return QPFunction(verbose=-1, **solver_args)(
            Qs.double(),
            ps.double(),
            Gs.double(),
            hs.double(),
            As,
            bs,
        ).float()

    # pylint: disable-next=too-many-locals
    def get_cbf_qp_constraints(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        mean_pred_batch: torch.Tensor,
        sigma_pred_batch: torch.Tensor,
        gamma_b: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds up matrices required to solve a quadratic program (QP).

        The QP is defined to solve:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
            subject to G[u,eps]^T <= h

        Args:
            state_batch (torch.Tensor): Current state batch.
            action_batch (torch.Tensor): Nominal control input batch.
            mean_pred_batch (torch.Tensor): Mean disturbance prediction state batch.
            sigma_pred_batch (torch.Tensor): Standard deviation of the additive disturbance.
            gamma_b (float, optional): CBF parameter for the class-Kappa function. Defaults to 1.0.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                P (torch.Tensor): Quadratic cost matrix in the QP.
                q (torch.Tensor): Linear cost vector in the QP.
                G (torch.Tensor): Inequality constraint matrix for QP constraints.
                h (torch.Tensor): Inequality constraint vector for QP constraints.
        """
        assert (
            len(state_batch.shape) == 2
            and len(action_batch.shape) == 2
            and len(mean_pred_batch.shape) == 2
            and len(sigma_pred_batch.shape) == 2
        ), print(
            state_batch.shape,
            action_batch.shape,
            mean_pred_batch.shape,
            sigma_pred_batch.shape,
        )

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        state_batch = torch.unsqueeze(state_batch, -1).to(self.device)
        action_batch = torch.unsqueeze(action_batch, -1).to(self.device)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1).to(self.device)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1).to(self.device)
        if self.env.dynamics_mode == 'Unicycle':

            num_cbfs = len(self.env.hazards)
            l_p = self.l_p
            buffer = 0.1

            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas
            f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)
            Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = -s_thetas
            Rs[:, 1, 0] = s_thetas
            Rs[:, 1, 1] = c_thetas
            Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
            Ls[:, 0, 0] = 1
            Ls[:, 1, 1] = l_p
            g_ps = torch.bmm(Rs, Ls)
            mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
            mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
            sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
            sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

            hs = 1e3 * torch.ones((batch_size, num_cbfs), device=self.device)
            dhdps = torch.zeros((batch_size, num_cbfs, 2), device=self.device)
            hazards = self.env.hazards
            for i, hazard in enumerate(hazards):
                if hazard['type'] == 'circle':
                    obs_loc = to_tensor(hazard['location'], torch.FloatTensor, self.device)
                    hs[:, i] = 0.5 * (
                        torch.sum((ps - obs_loc) ** 2, dim=1) - (hazard['radius'] + buffer) ** 2
                    )
                    dhdps[:, i, :] = ps - obs_loc
                else:
                    raise NotImplementedError

            n_u = action_batch.shape[1]
            num_constraints = num_cbfs + 2 * n_u

            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            G[:, :num_cbfs, :n_u] = -torch.bmm(dhdps, g_ps)
            G[:, :num_cbfs, n_u] = -1
            h[:, :num_cbfs] = gamma_b * (hs**3) + (
                torch.bmm(dhdps, f_ps + mu_ps)
                - torch.bmm(torch.abs(dhdps), sigma_ps)
                + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)
            ).squeeze(-1)
            ineq_constraint_counter += num_cbfs
            P = (
                torch.diag(torch.tensor([1.0e0, 1.0e-2, 1e5]))
                .repeat(batch_size, 1, 1)
                .to(self.device)
            )
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        n_u = action_batch.shape[1]

        for c in range(n_u):

            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the action bounds.

        Returns:
            Action bounds, i.e., min control input and max control input.
        """
        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max
