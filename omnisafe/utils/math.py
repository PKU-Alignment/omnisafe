# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the algo utils."""
from typing import Callable, Tuple

import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution


def get_transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a tensor.

    Args:
        tensor: torch.Tensor

    Returns:
        torch.Tensor
    """
    return tensor.transpose(dim0=-2, dim1=-1)


def get_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """Get the diagonal of the last two dimensions of a tensor.

    Args:
        tensor: torch.Tensor

    Returns:
        torch.Tensor
    """
    return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)


def safe_inverse(var_q: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
    """Inverse of a matrix with a safe guard for singular matrix.

    Args:
        var_q: torch.Tensor
        det: torch.Tensor

    Returns:
        torch.Tensor
    """
    indices = torch.where(det <= 1e-6)
    # pseudo inverse
    if len(indices[0]) > 0:
        return torch.linalg.pinv(var_q)
    return var_q.inverse()


def gaussian_kl(
    mean_p: torch.Tensor, mean_q: torch.Tensor, var_p: torch.Tensor, var_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decoupled KL between two mean_qltivariate gaussian distribution.

    .. note::
        Detailedly,

        .. math::
            KL(q||p) = 0.5 * (tr(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^T \Sigma_p^{-1} (\mu_p -
            \mu_q) - k + log(\frac{det(\Sigma_p)}{det(\Sigma_q)}))

        where :math:`\mu_p` and :math:`\mu_q` are the mean of :math:`p` and :math:`q`, respectively.
        :math:`\Sigma_p` and :math:`\Sigma_q` are the co-variance of :math:`p` and :math:`q`, respectively.
        :math:`k` is the dimension of the distribution.

    For more details,
    please refer to the paper `A General and Adaptive Robust Loss Function <https://arxiv.org/abs/1701.03077>`_,
    and the notes `here <https://stanford.edu/~jduchi/projects/general_notes.pdf>`_.

    Args:
        mean_p (torch.Tensor): mean of the first distribution, shape (B, n)
        mean_q (torch.Tensor): mean of the second distribution, shape (B, n)
        var_p (torch.Tensor): co-variance of the first distribution, shape (B, n, n)
        var_q (torch.Tensor): co-variance of the second distribution, shape (B, n, n)
    """
    len_q = var_q.size(-1)
    mean_p = mean_p.unsqueeze(-1)  # (B, n, 1)
    mean_q = mean_q.unsqueeze(-1)  # (B, n, 1)
    sigma_p = var_p @ get_transpose(var_p)  # (B, n, n)
    sigma_q = var_q @ get_transpose(var_q)  # (B, n, n)
    sigma_p_det = sigma_p.det()  # (B,)
    sigma_q_det = sigma_q.det()  # (B,)
    sigma_p_inv = safe_inverse(sigma_p, sigma_p_det)  # (B, n, n)
    sigma_q_inv = safe_inverse(sigma_q, sigma_q_det)  # (B, n, n)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    sigma_p_det = torch.clamp_min(sigma_p_det, 1e-6)
    sigma_q_det = torch.clamp_min(sigma_q_det, 1e-6)
    inner_mean_q = ((mean_q - mean_p).transpose(-2, -1) @ sigma_p_inv @ (mean_q - mean_p)).squeeze()
    inner_sigma_q = (
        torch.log(sigma_q_det / sigma_p_det) - len_q + get_diagonal(sigma_q_inv @ sigma_p)
    )
    c_mean_q = 0.5 * torch.mean(inner_mean_q)
    c_sigma_q = 0.5 * torch.mean(inner_sigma_q)
    return c_mean_q, c_sigma_q, torch.mean(sigma_p_det), torch.mean(sigma_q_det)


def discount_cumsum(x_vector: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors."""
    length = x_vector.shape[0]
    x_vector = x_vector.type(torch.float64)
    for idx in reversed(range(length)):
        if idx == length - 1:
            cumsum = x_vector[idx]
        else:
            cumsum = x_vector[idx] + discount * cumsum
        x_vector[idx] = cumsum
    return x_vector


def conjugate_gradients(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b_vector: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
):  # pylint: disable=invalid-name,too-many-locals
    """Implementation of Conjugate gradient algorithm.

    Conjugate gradient algorithm is used to solve the linear system of equations :math:`Ax = b`.
    The algorithm is described in detail in the paper `Conjugate Gradient Method`_.

    .. _Conjugate Gradient Method: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        Increasing ``num_steps`` will lead to a more accurate approximation
        to :math:`A^{-1} b`, and possibly slightly-improved performance,
        but at the cost of slowing things down.
        Also probably don't play with this hyperparameter.

    Args:
        num_steps (int): Number of iterations of conjugate gradient to perform.
    """

    x = torch.zeros_like(b_vector)
    r = b_vector - Avp(x)
    p = r.clone()
    rdotr = torch.dot(r, r)

    for _ in range(num_steps):
        z = Avp(p)
        alpha = rdotr / (torch.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr
    return x


class SafeTanhTransformer(TanhTransform):
    """Safe Tanh Transformer."""

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.tanh(x), min=-0.999999, max=0.999999)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).eps
        else:
            raise ValueError('Expected floating point type')
        y = y.clamp(min=-1 + eps, max=1 - eps)
        x = super()._inverse(y)
        return x


class TanhNormal(TransformedDistribution):  # pylint: disable=abstract-method
    r"""
    Creates a tanh-normal distribution.

        X ~ Normal(loc, scale)
        Y = tanh(X) ~ TanhNormal(loc, scale)

    Example::

        >>> m = TanhNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # tanh-normal distributed with mean=0 and stddev=1
        tensor([-0.7616])

    Args:
        loc (float or Tensor): mean of the underlying normal distribution
        scale (float or Tensor): standard deviation of the underlying normal distribution
    """

    arg_constraints = {
        'loc': Normal.arg_constraints['loc'],
        'scale': Normal.arg_constraints['scale'],
    }
    support = TransformedDistribution.support
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, SafeTanhTransformer(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return SafeTanhTransformer()(self.base_dist.mean)

    @property
    def stddev(self):
        return self.base_dist.stddev

    def entropy(self):
        return self.base_dist.entropy()

    @property
    def variance(self):
        return self.base_dist.variance
