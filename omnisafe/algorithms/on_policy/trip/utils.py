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
"""Linear-algebra helpers for TRIP-PO and TRIP-PPO."""

from __future__ import annotations

from collections.abc import Callable

import torch


MetricProduct = Callable[[torch.Tensor], torch.Tensor]


def damped_solve(matrix: torch.Tensor, rhs: torch.Tensor, damping: float) -> torch.Tensor:
    """Solve a small linear system after adding Tikhonov damping."""
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve(matrix + damping * identity, rhs)


def project_to_kernel(
    vector: torch.Tensor,
    jacobian: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Apply ``I - J.T (J J.T + damping I)^-1 J`` without forming the projector."""
    if jacobian.shape[0] == 0:
        return vector
    correction = jacobian.T @ damped_solve(
        jacobian @ jacobian.T,
        jacobian @ vector,
        damping,
    )
    return vector - correction


def scale_to_metric_radius(
    vector: torch.Tensor,
    metric_product: MetricProduct,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a vector so its squared metric norm is no larger than ``radius``."""
    norm_sq = torch.clamp(torch.dot(vector, metric_product(vector)), min=0.0)
    radius_tensor = torch.as_tensor(radius, dtype=vector.dtype, device=vector.device)
    scale = torch.minimum(
        torch.ones((), dtype=vector.dtype, device=vector.device),
        torch.sqrt(radius_tensor / (norm_sq + torch.finfo(vector.dtype).eps)),
    )
    return scale * vector, norm_sq


def normal_step(  # pylint: disable=too-many-arguments
    residual: torch.Tensor,
    constraint_jacobian: torch.Tensor,
    slack: torch.Tensor,
    metric_product: MetricProduct,
    radius: float,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the damped, radius-limited TRIP normal step.

    The second returned tensor is the scaled slack displacement ``Z^-1 delta_zeta``.
    """
    gamma = torch.cat((constraint_jacobian, torch.diag(slack)), dim=1)
    raw_step = -gamma.T @ damped_solve(gamma @ gamma.T, residual, damping)
    step, raw_norm_sq = scale_to_metric_radius(raw_step, metric_product, radius)
    policy_size = constraint_jacobian.shape[1]
    return step[:policy_size], step[policy_size:], raw_norm_sq
