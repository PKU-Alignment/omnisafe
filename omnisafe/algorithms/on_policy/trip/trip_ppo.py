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
"""Implementation of Trust-Region Interior-Point PPO."""

from __future__ import annotations

import copy

import torch
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.algorithms.on_policy.trip.mixin import TRIPMixin
from omnisafe.utils import distributed
from omnisafe.utils.tools import get_flat_params_from, set_param_values_to_model


@registry.register
class TRIPPPO(TRIPMixin, PPO):
    """TRIP-PPO for one cost constraint.

    A normal feasibility step is followed by PPO actor substeps.  Their accumulated displacement
    is projected once onto the kernel of the rollout's linearized cost constraint.
    """

    def _surrogate_constraint_value(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        constraint: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the candidate-policy surrogate constraint value."""
        self._actor_critic.actor(obs)
        ratio = torch.exp(self._actor_critic.actor.log_prob(act) - logp)
        cost_change = (ratio - 1.0) * adv_c
        cost_change = cost_change.mean() * self._trajectory_scale(obs)
        return constraint + cost_change

    def _surrogate_cost_residual(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        constraint: torch.Tensor,
        slack: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the candidate-policy constraint residual including slack."""
        constraint_at_candidate = self._surrogate_constraint_value(
            obs,
            act,
            logp,
            adv_c,
            constraint,
        )
        return constraint_at_candidate + slack

    def _update(self) -> None:  # pylint: disable=too-many-locals
        """Update the actor with normal plus projected PPO steps and train both critics."""
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        theta_old = get_flat_params_from(self._actor_critic.actor)
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)

        constraint_jacobian = self._constraint_jacobian(obs, act, logp, adv_c)
        joint_transposes, joint_schur = self._kernel_basis(constraint_jacobian)
        policy_size = theta_old.numel()
        policy_transposes = constraint_jacobian
        damping = float(self._cfgs.algo_cfgs.trip_damping)
        policy_schur = constraint_jacobian @ constraint_jacobian.T + damping * torch.eye(
            constraint_jacobian.shape[0],
            dtype=constraint_jacobian.dtype,
            device=constraint_jacobian.device,
        )
        slack_old = self._slack.clone()
        constraint = self._constraint_value()
        normal_theta, normal_slack, _ = self._normal_step(
            constraint_jacobian,
            constraint,
            (joint_transposes, joint_schur),
        )
        # Commit the normal feasibility step before generating tangential PPO steps.
        set_param_values_to_model(self._actor_critic.actor, theta_old + normal_theta)
        actor_optimizer_state_before_ppo = copy.deepcopy(
            self._actor_critic.actor_optimizer.state_dict(),
        )
        dataloader = DataLoader(
            TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        update_counts = 0
        ppo_kl = 0.0
        max_substep_projection_error = 0.0
        unprojected_tangent_norm_sq = 0.0
        for iteration in track(
            range(self._cfgs.algo_cfgs.update_iters),
            description='Generating projected PPO steps...',
        ):
            for batch in dataloader:
                (
                    batch_obs,
                    batch_act,
                    batch_logp,
                    batch_target_r,
                    batch_target_c,
                    batch_adv_r,
                    batch_adv_c,
                ) = batch
                self._update_reward_critic(batch_obs, batch_target_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(batch_obs, batch_target_c)
                theta_before_substep = get_flat_params_from(self._actor_critic.actor)
                self._update_actor(
                    batch_obs,
                    batch_act,
                    batch_logp,
                    batch_adv_r,
                    batch_adv_c,
                )
                # Keep the raw Adam/PPO trajectory intact.  Projection is applied once to the
                # aggregate displacement after all minibatches, preserving optimizer coherence.
                unprojected_substep = (
                    get_flat_params_from(self._actor_critic.actor) - theta_before_substep
                )
                unprojected_tangent_norm_sq += torch.dot(
                    unprojected_substep,
                    unprojected_substep,
                ).item()

            with torch.no_grad():
                ppo_distribution = self._actor_critic.actor(obs)
                ppo_kl_tensor = (
                    torch.distributions.kl.kl_divergence(old_distribution, ppo_distribution)
                    .sum(-1, keepdim=True)
                    .mean()
                )
                ppo_kl = distributed.dist_avg(ppo_kl_tensor).item()
            update_counts += 1
            if self._cfgs.algo_cfgs.kl_early_stop and ppo_kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(
                    f'Early stopping PPO displacement at iter {iteration + 1} '
                    'due to reaching max kl',
                )
                break

        raw_tangent_theta = (
            get_flat_params_from(self._actor_critic.actor) - theta_old - normal_theta
        )
        raw_tangent_joint = torch.cat((raw_tangent_theta, torch.zeros_like(self._slack)))
        raw_constraint_norm = torch.linalg.vector_norm(constraint_jacobian @ raw_tangent_theta)
        if self._cfgs.algo_cfgs.project_tangential_substeps:
            # Project only the PPO policy displacement with the Euclidean projector.
            # The tangential PPO step has no slack component.
            tangent_theta = self._project_to_kernel(
                raw_tangent_theta,
                constraint_jacobian,
                policy_transposes,
                policy_schur,
            )
            tangent_slack = torch.zeros_like(self._slack)
            tangent_joint = torch.cat((tangent_theta, tangent_slack))
            set_param_values_to_model(
                self._actor_critic.actor,
                theta_old + normal_theta + tangent_theta,
            )
            projected_constraint_norm = torch.linalg.vector_norm(
                constraint_jacobian @ tangent_theta,
            )
            projection_constraint_ratio = projected_constraint_norm / (
                raw_constraint_norm + torch.finfo(raw_constraint_norm.dtype).eps
            )
        else:
            # Ablation mode: commit the raw PPO/Adam displacement directly.
            tangent_joint = raw_tangent_joint
            tangent_theta = tangent_joint[: theta_old.numel()]
            tangent_slack = tangent_joint[theta_old.numel() :]
            projection_constraint_ratio = torch.ones(
                (), dtype=raw_constraint_norm.dtype, device=raw_constraint_norm.device,
            )
        slack_after_normal = self._slack + self._slack * normal_slack

        # Backtrack the accumulated tangential displacement using only the existing KL bound.
        # The nonlinear residual is retained as a diagnostic, but is not an acceptance gate.
        nonlinear_alpha = 1.0
        nonlinear_evaluations = 0
        with torch.no_grad():
            set_param_values_to_model(self._actor_critic.actor, theta_old + normal_theta)
            normal_residual = self._surrogate_cost_residual(
                obs,
                act,
                logp,
                adv_c,
                constraint,
                slack_after_normal,
            )

            accepted = not self._cfgs.algo_cfgs.nonlinear_filter
            if accepted:
                set_param_values_to_model(
                    self._actor_critic.actor,
                    theta_old + normal_theta + tangent_theta,
                )
            else:
                filter_steps = max(1, int(self._cfgs.algo_cfgs.nonlinear_filter_steps))
                for step in range(filter_steps):
                    alpha = float(self._cfgs.algo_cfgs.nonlinear_filter_decay) ** step
                    set_param_values_to_model(
                        self._actor_critic.actor,
                        theta_old + normal_theta + alpha * tangent_theta,
                    )
                    candidate_distribution = self._actor_critic.actor(obs)
                    candidate_kl = torch.distributions.kl.kl_divergence(
                        old_distribution,
                        candidate_distribution,
                    ).sum(-1, keepdim=True).mean()
                    nonlinear_evaluations += 1
                    if (
                        torch.isfinite(candidate_kl).all()
                        and candidate_kl <= self._cfgs.algo_cfgs.target_kl
                    ):
                        nonlinear_alpha = alpha
                        accepted = True
                        break

            if not accepted:
                nonlinear_alpha = 0.0
                set_param_values_to_model(self._actor_critic.actor, theta_old + normal_theta)

        if self._cfgs.algo_cfgs.project_tangential_substeps or nonlinear_alpha != 1.0:
            # Adam's moments describe the raw trajectory.  Discard them whenever projection or
            # nonlinear filtering changes the trajectory that is actually committed.
            self._actor_critic.actor_optimizer.load_state_dict(actor_optimizer_state_before_ppo)
            tangent_joint = nonlinear_alpha * tangent_joint
            tangent_theta = tangent_joint[: theta_old.numel()]
            tangent_slack = tangent_joint[theta_old.numel() :]

        # Evaluate c(theta_c) with a fresh environment rollout and apply the joint
        # Euclidean SOC correction. The slack coordinate remains scaled, so the
        # correction is represented as (d_theta_soc, Z^-1 d_zeta_soc).
        slack_after_tangent = self._slack + self._slack * (
            normal_slack + tangent_slack
        )
        soc_enabled = bool(getattr(self._cfgs.algo_cfgs, 'soc_enabled', False))
        if soc_enabled:
            soc_rollout_episodes = int(
                getattr(self._cfgs.algo_cfgs, 'soc_rollout_episodes', 1),
            )
            with torch.no_grad():
                candidate_cost = self._env.evaluate_cost(
                    self._actor_critic,
                    num_episodes=soc_rollout_episodes,
                )
                # Use the full post-candidate residual as the SOC right-hand side.
                soc_residual = (
                    candidate_cost
                    - float(self._cfgs.algo_cfgs.cost_limit)
                    + slack_after_tangent
                )
        else:
            soc_residual = torch.zeros_like(self._slack)
        soc_joint = torch.zeros(
            theta_old.numel() + self._slack.numel(),
            dtype=theta_old.dtype,
            device=theta_old.device,
        )
        if soc_enabled and soc_residual.item() > 0.0:
            soc_joint = self._second_order_correction(
                constraint_jacobian,
                soc_residual,
                slack_after_tangent,
            )
        soc_theta = soc_joint[: theta_old.numel()]
        soc_slack = soc_joint[theta_old.numel() :]
        if soc_enabled and soc_residual.item() > 0.0:
            theta_before_soc = get_flat_params_from(self._actor_critic.actor)
            set_param_values_to_model(
                self._actor_critic.actor,
                theta_before_soc + soc_theta,
            )
        final_slack = torch.clamp(
            slack_after_tangent + slack_after_tangent * soc_slack,
            min=float(self._cfgs.algo_cfgs.slack_min),
        ).detach()
        self._slack = final_slack

        with torch.no_grad():
            final_distribution = self._actor_critic.actor(obs)
            final_kl_tensor = (
                torch.distributions.kl.kl_divergence(old_distribution, final_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            final_kl = distributed.dist_avg(final_kl_tensor).item()

        projection_error = torch.linalg.vector_norm(constraint_jacobian @ tangent_theta)
        with torch.no_grad():
            accepted_residual = self._surrogate_cost_residual(
                obs,
                act,
                logp,
                adv_c,
                constraint,
                final_slack,
            )
        linearized_residual = (
            constraint
            + slack_old
            + constraint_jacobian @ (normal_theta + tangent_theta)
            + slack_old * normal_slack
        )
        self._logger.store(
            {
                'Train/StopIter': update_counts,
                'Train/KL': final_kl,
                'Value/Adv': adv_r.mean().item(),
                'Misc/TRIPSlack': self._slack.item(),
                'Misc/TRIPResidual': linearized_residual.item(),
                'Misc/TRIPNormalStepNorm': torch.linalg.vector_norm(normal_theta).item(),
                'Misc/TRIPTangentialStepNorm': torch.linalg.vector_norm(tangent_theta).item(),
                'Misc/TRIPProjectionError': projection_error.item(),
                'Misc/TRIPSubstepProjectionError': max_substep_projection_error,
                'Misc/TRIPAggregateProjectionError': projection_error.item(),
                'Misc/TRIPProjectionConstraintRatio': projection_constraint_ratio.item(),
                'Misc/TRIPUnprojectedTangentialStepNorm': unprojected_tangent_norm_sq**0.5,
                'Misc/TRIPCostEstimate': constraint.item() + self._cfgs.algo_cfgs.cost_limit,
                'Misc/TRIPNonlinearAcceptanceAlpha': nonlinear_alpha,
                'Misc/TRIPNonlinearResidualNormal': normal_residual.item(),
                'Misc/TRIPNonlinearResidualAccepted': accepted_residual.item(),
                'Misc/TRIPNonlinearFilterEvaluations': nonlinear_evaluations,
                'Misc/TRIPSOCResidual': soc_residual.item(),
                'Misc/TRIPSOCStepNorm': torch.linalg.vector_norm(soc_theta).item(),
                'Misc/TRIPSOCSlackStep': soc_slack.item(),
            },
        )
        self._logger.log(f'Projected PPO KL: {ppo_kl:.6f}; final TRIP-PPO KL: {final_kl:.6f}')
