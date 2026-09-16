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
"""Implementation of Trust-Region Interior-Point Policy Optimization."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.algorithms.on_policy.trip.mixin import TRIPMixin
from omnisafe.algorithms.on_policy.trip.utils import project_to_kernel, scale_to_metric_radius
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class TRIPPO(TRIPMixin, TRPO):
    """Trust-Region Interior-Point Policy Optimization for one cost constraint."""

    _barrier: float

    def _init(self) -> None:
        super()._init()
        self._barrier = float(self._cfgs.algo_cfgs.barrier_coefficient)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/TRIPBarrier')

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Apply an unsearched normal step and a TRPO-searched tangential step."""
        theta_old = get_flat_params_from(self._actor_critic.actor)
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)

        constraint_jacobian = self._constraint_jacobian(obs, act, logp, adv_c)
        slack_old = self._slack.clone()
        constraint = self._constraint_value()
        normal_theta, normal_slack, _ = self._normal_step(constraint_jacobian, constraint)

        self._actor_critic.actor.zero_grad()
        reward_loss = self._loss_pi(obs, act, logp, adv_r) * self._trajectory_scale(obs)
        reward_loss.backward()
        distributed.avg_grads(self._actor_critic.actor)
        objective_gradient = get_flat_gradients_from(self._actor_critic.actor).detach().clone()

        gamma = torch.cat((constraint_jacobian, torch.diag(self._slack)), dim=1)
        barrier_gradient = torch.full_like(self._slack, -self._barrier)
        joint_gradient = torch.cat((objective_gradient, barrier_gradient))
        projected_descent = -project_to_kernel(
            joint_gradient,
            gamma,
            float(self._cfgs.algo_cfgs.trip_damping),
        )

        def projected_metric_product(vector: torch.Tensor) -> torch.Tensor:
            projected_vector = project_to_kernel(
                vector,
                gamma,
                float(self._cfgs.algo_cfgs.trip_damping),
            )
            return project_to_kernel(
                self._joint_metric_product(projected_vector),
                gamma,
                float(self._cfgs.algo_cfgs.trip_damping),
            )

        unconstrained_tangent = conjugate_gradients(
            projected_metric_product,
            projected_descent,
            self._cfgs.algo_cfgs.cg_iters,
        )
        raw_tangent = project_to_kernel(
            unconstrained_tangent,
            gamma,
            float(self._cfgs.algo_cfgs.trip_damping),
        )
        tangent_radius = min(
            float(self._cfgs.algo_cfgs.tangential_step_radius),
            0.5
            * (1.0 - float(self._cfgs.algo_cfgs.xi))
            * float(self._cfgs.algo_cfgs.fraction_to_boundary) ** 2,
        )
        tangent, raw_tangent_norm_sq = scale_to_metric_radius(
            raw_tangent,
            self._joint_metric_product,
            tangent_radius,
        )
        tangent_theta = tangent[: theta_old.numel()]
        tangent_slack = tangent[theta_old.numel() :]

        theta_after_normal = theta_old + normal_theta
        set_param_values_to_model(self._actor_critic.actor, theta_after_normal)
        with torch.no_grad():
            loss_after_normal = distributed.dist_avg(self._loss_pi(obs, act, logp, adv_r))

        accepted_tangent, acceptance_step = self._search_step_size(
            step_direction=tangent_theta,
            grads=-objective_gradient,
            p_dist=old_distribution,
            obs=obs,
            act=act,
            logp=logp,
            adv=adv_r,
            loss_before=loss_after_normal,
            total_steps=int(self._cfgs.algo_cfgs.line_search_steps),
            decay=float(self._cfgs.algo_cfgs.line_search_decay),
        )
        line_scale = (
            0.0
            if acceptance_step == 0
            else float(self._cfgs.algo_cfgs.line_search_decay) ** (acceptance_step - 1)
        )
        set_param_values_to_model(
            self._actor_critic.actor,
            theta_after_normal + accepted_tangent,
        )
        self._update_slack(normal_slack + line_scale * tangent_slack)

        projection_error = torch.linalg.vector_norm(gamma @ (line_scale * tangent))
        joint_step = torch.cat(
            (normal_theta + accepted_tangent, normal_slack + line_scale * tangent_slack),
        )
        linearized_residual = constraint + slack_old + gamma @ joint_step
        self._logger.store(
            {
                'Misc/AcceptanceStep': acceptance_step,
                'Misc/Alpha': line_scale,
                'Misc/FinalStepNorm': torch.linalg.vector_norm(
                    normal_theta + accepted_tangent,
                ).item(),
                'Misc/xHx': raw_tangent_norm_sq.item(),
                'Misc/H_inv_g': torch.linalg.vector_norm(raw_tangent).item(),
                'Misc/gradient_norm': torch.linalg.vector_norm(objective_gradient).item(),
                'Misc/TRIPSlack': self._slack.item(),
                'Misc/TRIPResidual': linearized_residual.item(),
                'Misc/TRIPNormalStepNorm': torch.linalg.vector_norm(normal_theta).item(),
                'Misc/TRIPTangentialStepNorm': torch.linalg.vector_norm(accepted_tangent).item(),
                'Misc/TRIPProjectionError': projection_error.item(),
                'Misc/TRIPBarrier': self._barrier,
                'Misc/TRIPCostEstimate': constraint.item() + self._cfgs.algo_cfgs.cost_limit,
            },
        )
        self._barrier = max(
            float(self._cfgs.algo_cfgs.barrier_min),
            self._barrier * float(self._cfgs.algo_cfgs.barrier_decay),
        )
