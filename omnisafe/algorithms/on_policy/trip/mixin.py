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
"""Shared state and derivatives for the TRIP algorithms."""

from __future__ import annotations

from collections import deque

import torch

from omnisafe.algorithms.on_policy.trip.utils import scale_to_metric_radius
from omnisafe.utils import distributed
from omnisafe.utils.tools import get_flat_gradients_from


class TRIPMixin:  # pylint: disable=too-few-public-methods
    """Provide scalar-cost TRIP initialization and normal-step computation."""

    _slack: torch.Tensor

    def _init(self) -> None:
        super()._init()
        self._slack = torch.full(
            (1,),
            float(self._cfgs.algo_cfgs.slack_init),
            device=self._device,
        )
        self._cost_metric = self._cfgs.algo_cfgs.cost_metric
        if self._cost_metric == 'window':
            self._cost_metric = 'reporting_window'
        if self._cost_metric not in ('current', 'rollout_window', 'reporting_window'):
            raise ValueError(
                "cost_metric must be 'current', 'rollout_window', or 'reporting_window', "
                f'got {self._cost_metric!r}.',
            )
        cost_window_size = int(self._cfgs.algo_cfgs.cost_window_size)
        jacobian_window_size = int(self._cfgs.algo_cfgs.jacobian_window_size)
        if cost_window_size < 1 or jacobian_window_size < 1:
            raise ValueError('cost_window_size and jacobian_window_size must both be positive.')
        self._cost_history: deque[float] = deque(maxlen=cost_window_size)
        self._jacobian_history: deque[torch.Tensor] = deque(maxlen=jacobian_window_size)
        self._previous_constraint_jacobian: torch.Tensor | None = None

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/TRIPSlack')
        self._logger.register_key('Misc/TRIPResidual')
        self._logger.register_key('Misc/TRIPNormalStepNorm')
        self._logger.register_key('Misc/TRIPTangentialStepNorm')
        self._logger.register_key('Misc/TRIPProjectionError')
        self._logger.register_key('Misc/TRIPSubstepProjectionError')
        self._logger.register_key('Misc/TRIPAggregateProjectionError')
        self._logger.register_key('Misc/TRIPUnprojectedTangentialStepNorm')
        self._logger.register_key('Misc/TRIPNonlinearAcceptanceAlpha')
        self._logger.register_key('Misc/TRIPNonlinearResidualNormal')
        self._logger.register_key('Misc/TRIPNonlinearResidualAccepted')
        self._logger.register_key('Misc/TRIPNonlinearFilterEvaluations')
        self._logger.register_key('Misc/TRIPNumTrajectories')
        self._logger.register_key('Misc/TRIPConstraintJacobianNorm')
        self._logger.register_key('Misc/TRIPCurrentJacobianNorm')
        self._logger.register_key('Misc/TRIPJacobianWindowSize')
        self._logger.register_key('Misc/TRIPJacobianCosine')
        self._logger.register_key('Misc/TRIPProjectionConstraintRatio')
        self._logger.register_key('Misc/TRIPCostEstimate')
        self._logger.register_key('Misc/TRIPSOCResidual')
        self._logger.register_key('Misc/TRIPSOCStepNorm')
        self._logger.register_key('Misc/TRIPSOCSlackStep')

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Return the trajectory-averaged cost surrogate used to estimate its Jacobian."""
        self._actor_critic.actor(obs)
        new_logp = self._actor_critic.actor.log_prob(act)
        return (torch.exp(new_logp - logp) * adv_c).mean() * self._trajectory_scale(obs)

    def _trajectory_scale(self, obs: torch.Tensor) -> float:
        """Convert a transition mean into a mean of per-trajectory timestep sums."""
        num_trajectories = self._buf.last_num_paths
        if num_trajectories <= 0:
            raise RuntimeError('TRIP requires at least one completed or bootstrapped trajectory.')
        mean_num_trajectories = torch.tensor(
            [float(num_trajectories)],
            dtype=obs.dtype,
            device=obs.device,
        )
        distributed.avg_tensor(mean_num_trajectories)
        return obs.shape[0] / mean_num_trajectories.item()

    def _constraint_value(self) -> torch.Tensor:
        """Return the configured cost estimate minus the episodic constraint limit."""
        current_cost = self._logger.get_stats('Metrics/CurrentEpCost')[0]
        if self._cost_metric == 'current':
            cost_estimate = current_cost
        elif self._cost_metric == 'reporting_window':
            cost_estimate = self._logger.get_stats('Metrics/EpCost')[0]
        else:
            self._cost_history.append(current_cost)
            cost_estimate = sum(self._cost_history) / len(self._cost_history)
        return torch.as_tensor(
            cost_estimate - self._cfgs.algo_cfgs.cost_limit,
            dtype=self._slack.dtype,
            device=self._slack.device,
        ).reshape(1)

    def _constraint_jacobian(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the single-constraint policy Jacobian as a one-row matrix."""
        self._actor_critic.actor.zero_grad()
        self._loss_pi_cost(obs, act, logp, adv_c).backward()
        distributed.avg_grads(self._actor_critic.actor)
        current_gradient = get_flat_gradients_from(self._actor_critic.actor).detach().clone()
        current_norm = torch.linalg.vector_norm(current_gradient)
        if self._previous_constraint_jacobian is None:
            jacobian_cosine = 1.0
        else:
            previous_norm = torch.linalg.vector_norm(self._previous_constraint_jacobian)
            jacobian_cosine = torch.dot(current_gradient, self._previous_constraint_jacobian) / (
                current_norm * previous_norm + torch.finfo(current_gradient.dtype).eps
            )
        self._previous_constraint_jacobian = current_gradient.clone()
        self._jacobian_history.append(current_gradient)
        gradient = torch.stack(tuple(self._jacobian_history)).mean(dim=0)
        self._logger.store(
            {
                'Misc/TRIPNumTrajectories': self._buf.last_num_paths,
                'Misc/TRIPConstraintJacobianNorm': torch.linalg.vector_norm(gradient).item(),
                'Misc/TRIPCurrentJacobianNorm': current_norm.item(),
                'Misc/TRIPJacobianWindowSize': len(self._jacobian_history),
                'Misc/TRIPJacobianCosine': float(jacobian_cosine),
            },
        )
        return gradient.unsqueeze(0)

    def _joint_metric_product(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the Euclidean metric to a joint vector."""
        return vector

    def _kernel_basis(
        self,
        jacobian: torch.Tensor,
        slack: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the Euclidean constraint basis and damped Schur complement."""
        if slack is None:
            slack = self._slack
        gamma = torch.cat((jacobian, torch.diag(slack)), dim=1)
        damping = float(self._cfgs.algo_cfgs.trip_damping)
        schur = gamma @ gamma.T + damping * torch.eye(
            gamma.shape[0], dtype=gamma.dtype, device=gamma.device,
        )
        # Keep the same orientation used by the projection helpers:
        # constraints x joint_variables.
        return gamma, schur

    @staticmethod
    def _project_to_kernel(
        vector: torch.Tensor,
        jacobian: torch.Tensor,
        constraint_transposes: torch.Tensor,
        schur: torch.Tensor,
    ) -> torch.Tensor:
        """Project a policy step into ``ker(G)`` using the Euclidean metric."""
        if jacobian.shape[0] == 0:
            return vector
        multipliers = torch.linalg.solve(schur, jacobian @ vector)
        return vector - constraint_transposes.T @ multipliers

    def _second_order_correction(
        self,
        constraint_jacobian: torch.Tensor,
        residual: torch.Tensor,
        slack: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the joint Euclidean normal correction for a post-step residual."""
        joint_transposes, joint_schur = self._kernel_basis(
            constraint_jacobian,
            slack=slack,
        )
        multipliers = torch.linalg.solve(joint_schur, residual)
        return -joint_transposes.T @ multipliers

    def _normal_step(
        self,
        constraint_jacobian: torch.Tensor,
        constraint: torch.Tensor,
        joint_basis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a Euclidean minimum-norm normal step without line search.

        This solves ``min 1/2 v^T M_k v`` subject to ``Gamma v = -residual``.  The
        The policy and scaled-slack blocks both use the Euclidean metric.
        """
        residual = constraint + self._slack
        radius = min(
            float(self._cfgs.algo_cfgs.normal_step_radius),
            0.5
            * float(self._cfgs.algo_cfgs.xi)
            * float(self._cfgs.algo_cfgs.fraction_to_boundary) ** 2,
        )
        if joint_basis is None:
            joint_basis = self._kernel_basis(constraint_jacobian)
        joint_transposes, normal_schur = joint_basis
        multipliers = torch.linalg.solve(normal_schur, residual)
        raw_step = -joint_transposes.T @ multipliers
        policy_size = constraint_jacobian.shape[1]
        step, raw_norm_sq = scale_to_metric_radius(
            raw_step,
            self._joint_metric_product,
            radius,
        )
        policy_size = constraint_jacobian.shape[1]
        return step[:policy_size], step[policy_size:], raw_norm_sq

    def _update_slack(self, scaled_slack_step: torch.Tensor) -> None:
        """Update the slack and guard only against floating-point boundary error."""
        self._slack = torch.clamp(
            self._slack + self._slack * scaled_slack_step,
            min=float(self._cfgs.algo_cfgs.slack_min),
        ).detach()
