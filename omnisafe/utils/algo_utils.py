# Copyright 2022 OmniSafe Team. All Rights Reserved.
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

import torch


def bt(m: torch.tensor):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m: torch.tensor):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def safe_inverse(A, det):
    indices = torch.where(det <= 1e-6)
    # pseudoinverse
    if len(indices[0]) > 0:
        return torch.linalg.pinv(A)
    return A.inverse()


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinants of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_det = Σi.det()  # (B,)
    Σ_det = Σ.det()  # (B,)
    Σi_inv = safe_inverse(Σi, Σi_det)  # (B, n, n)
    Σ_inv = safe_inverse(Σ, Σ_det)  # (B, n, n)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    Σi_det = torch.clamp_min(Σi_det, 1e-6)
    Σ_det = torch.clamp_min(Σ_det, 1e-6)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)
