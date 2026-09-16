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
"""Tests for the shared TRIP linear algebra."""

import importlib.util
from pathlib import Path

import torch


_UTILS_PATH = (
    Path(__file__).parents[1] / 'omnisafe' / 'algorithms' / 'on_policy' / 'trip' / 'utils.py'
)
_SPEC = importlib.util.spec_from_file_location('trip_utils', _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)
normal_step = _UTILS.normal_step
project_to_kernel = _UTILS.project_to_kernel


def test_project_to_kernel() -> None:
    """Projection removes the component along a full-rank constraint Jacobian."""
    jacobian = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    vector = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)

    projected = project_to_kernel(vector, jacobian, damping=1e-12)

    assert torch.linalg.vector_norm(jacobian @ projected) < 1e-10


def test_project_to_kernel_with_dependent_constraints_is_finite() -> None:
    """Damping makes the projection well defined for dependent constraints."""
    jacobian = torch.tensor(
        [[1.0, 2.0, -1.0], [2.0, 4.0, -2.0]],
        dtype=torch.float64,
    )
    vector = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)

    projected = project_to_kernel(vector, jacobian, damping=1e-6)

    assert torch.isfinite(projected).all()
    assert torch.linalg.vector_norm(jacobian @ projected) < torch.linalg.vector_norm(
        jacobian @ vector,
    )


def test_normal_step_reduces_residual_and_obeys_radius() -> None:
    """The normal step improves feasibility while respecting its metric radius."""
    jacobian = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    slack = torch.tensor([0.5], dtype=torch.float64)
    residual = torch.tensor([2.0], dtype=torch.float64)
    radius = 0.05

    policy_step, scaled_slack_step, _ = normal_step(
        residual=residual,
        constraint_jacobian=jacobian,
        slack=slack,
        metric_product=lambda vector: vector,
        radius=radius,
        damping=1e-6,
    )
    joint_step = torch.cat((policy_step, scaled_slack_step))
    updated_residual = residual + jacobian @ policy_step + slack * scaled_slack_step

    assert torch.linalg.vector_norm(updated_residual) < torch.linalg.vector_norm(residual)
    assert torch.dot(joint_step, joint_step) <= radius + 1e-12
    assert torch.all(slack + slack * scaled_slack_step > 0.0)
