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

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution, constraints


def get_transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a tensor.

    Example:
        >>> tensor = torch.rand(2, 3, 4)
        >>> get_transpose(tensor).shape
        torch.Size([2, 4, 3])

    Args:
        tensor: torch.Tensor
    """
    return tensor.transpose(dim0=-2, dim1=-1)


def get_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """Get the diagonal of the last two dimensions of a tensor.

    Example:
        >>> tensor = torch.rand(2, 3, 4)
        >>> get_diagonal(tensor).shape
        torch.Size([2, 3])

    Args:
        tensor: torch.Tensor
    """
    return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)


def discount_cumsum(x_vector: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors.

    Example:
        >>> x_vector = torch.arange(1, 5)
        >>> x_vector
        tensor([1, 2, 3, 4])
        >>> discount_cumsum(x_vector, 0.9)
        tensor([4.00, 3.90, 3.00, 1.00])

    Args:
        x_vector (torch.Tensor): shape (B, T).
        discount (float): discount factor.
    """
    length = x_vector.shape[0]
    x_vector = x_vector.type(torch.float64)
    cumsum = x_vector[-1]
    for idx in reversed(range(length - 1)):
        cumsum = x_vector[idx] + discount * cumsum
        x_vector[idx] = cumsum
    return x_vector


def conjugate_gradients(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b_vector: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:  # pylint: disable=invalid-name,too-many-locals
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
        Avp (Callable[[torch.Tensor], torch.Tensor]):  Fisher information matrix vector product.
        b_vector (torch.Tensor): The vector :math:`b` in the equation :math:`Ax = b`.
        num_steps (int): The number of steps to run the algorithm for.
        residual_tol (float): The tolerance for the residual.
        eps (float): A small number to avoid dividing by zero.
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
    """Safe Tanh Transformer.

    This transformer is used to avoid the error caused by the input of tanh function
    being too large or too small.
    """

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.tanh(x), min=-0.999999, max=0.999999)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).eps
        else:
            raise ValueError('Expected floating point type')
        y = y.clamp(min=-1 + eps, max=1 - eps)
        return super()._inverse(y)


class TanhNormal(TransformedDistribution):  # pylint: disable=abstract-method
    r"""
    Creates a tanh-normal distribution.

    .. math::

        X \sim Normal(loc, scale)

        Y = tanh(X) \sim TanhNormal(loc, scale)

    Example::

        >>> m = TanhNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # tanh-normal distributed with mean=0 and stddev=1
        tensor([-0.7616])

    Args:
        loc (float or Tensor): mean of the underlying normal distribution
        scale (float or Tensor): standard deviation of the underlying normal distribution
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, SafeTanhTransformer())

    def expand(self, batch_shape: tuple, instance: Any | None = None) -> TanhNormal:
        """Expand the distribution."""
        new = self._get_checked_instance(TanhNormal, instance)
        return super().expand(batch_shape, new)

    @property
    def loc(self) -> torch.Tensor:
        """The loc of the tanh normal distribution."""
        return self.base_dist.mean

    @property
    def scale(self) -> torch.Tensor:
        """The scale of the tanh normal distribution."""
        return self.base_dist.stddev

    @property
    def mean(self) -> torch.Tensor:
        """The mean of the tanh normal distribution."""
        return SafeTanhTransformer()._call(self.base_dist.mean)

    @property
    def stddev(self) -> torch.Tensor:
        """The stddev of the tanh normal distribution."""
        return self.base_dist.stddev

    def entropy(self) -> torch.Tensor:
        """The entropy of the tanh normal distribution."""
        return self.base_dist.entropy()

    @property
    def variance(self) -> torch.Tensor:
        """The variance of the tanh normal distribution."""
        return self.base_dist.variance
