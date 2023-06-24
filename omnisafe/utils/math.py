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
"""Implementation of the math utils."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution, constraints


def get_transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a tensor.

    Examples:
        >>> tensor = torch.rand(2, 3)
        >>> get_transpose(tensor).shape
        torch.Size([3, 2])

    Args:
        tensor(torch.Tensor): The tensor to transpose.

    Returns:
        Transposed tensor.
    """
    return tensor.transpose(dim0=-2, dim1=-1)


def get_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """Get the diagonal of the last two dimensions of a tensor.

    Examples:
        >>> tensor = torch.rand(3, 3)
        >>> get_diagonal(tensor).shape
        torch.Size([1, 3])

    Args:
        tensor (torch.Tensor): The tensor to get the diagonal from.

    Returns:
        Diagonal part of the tensor.
    """
    return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)


def discount_cumsum(vector_x: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors.

    Examples:
        >>> vector_x = torch.arange(1, 5)
        >>> vector_x
        tensor([1, 2, 3, 4])
        >>> discount_cumsum(vector_x, 0.9)
        tensor([8.15, 5.23, 2.80, 1.00])

    Args:
        vector_x (torch.Tensor): A sequence of shape (B, T).
        discount (float): The discount factor.

    Returns:
        The discounted cumulative sum of vectors.
    """
    length = vector_x.shape[0]
    vector_x = vector_x.type(torch.float64)
    cumsum = vector_x[-1]
    for idx in reversed(range(length - 1)):
        cumsum = vector_x[idx] + discount * cumsum
        vector_x[idx] = cumsum
    return vector_x


# pylint: disable-next=too-many-locals
def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Implementation of Conjugate gradient algorithm.

    Conjugate gradient algorithm is used to solve the linear system of equations :math:`A x = b`.
    The algorithm is described in detail in the paper `Conjugate Gradient Method`_.

    .. _Conjugate Gradient Method: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        Increasing ``num_steps`` will lead to a more accurate approximation to :math:`A^{-1} b`, and
        possibly slightly-improved performance, but at the cost of slowing things down. Also
        probably don't play with this hyperparameter.

    Args:
        fisher_product (Callable[[torch.Tensor], torch.Tensor]): Fisher information matrix vector
            product.
        vector_b (torch.Tensor): The vector :math:`b` in the equation :math:`A x = b`.
        num_steps (int, optional): The number of steps to run the algorithm for. Defaults to 10.
        residual_tol (float, optional): The tolerance for the residual. Defaults to 1e-10.
        eps (float, optional): A small number to avoid dividing by zero. Defaults to 1e-6.

    Returns:
        The vector x in the equation Ax=b.
    """
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


class SafeTanhTransformer(TanhTransform):
    """Safe Tanh Transformer.

    This transformer is used to avoid the error caused by the input of tanh function being too large
    or too small.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the input."""
        return torch.clamp(torch.tanh(x), min=-0.999999, max=0.999999)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).eps
        else:
            raise ValueError('Expected floating point type')
        y = y.clamp(min=-1 + eps, max=1 - eps)
        return super()._inverse(y)


class TanhNormal(TransformedDistribution):  # pylint: disable=abstract-method
    r"""Create a tanh-normal distribution.

    .. math::

        X \sim Normal(loc, scale)

        Y = tanh(X) \sim TanhNormal(loc, scale)

    Examples:
        >>> m = TanhNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # tanh-normal distributed with mean=0 and stddev=1
        tensor([-0.7616])

    Args:
        loc (float or Tensor): The mean of the underlying normal distribution.
        scale (float or Tensor): The standard deviation of the underlying normal distribution.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        """Initialize an instance of :class:`TanhNormal`."""
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, SafeTanhTransformer())
        self.arg_constraints = {
            'loc': constraints.real,
            'scale': constraints.positive,
        }

    def expand(self, batch_shape: tuple[int, ...], instance: Any | None = None) -> TanhNormal:
        """Expand the distribution."""
        new = self._get_checked_instance(TanhNormal, instance)
        return super().expand(batch_shape, new)

    @property
    def loc(self) -> torch.Tensor:
        """The mean of the normal distribution."""
        return self.base_dist.mean

    @property
    def scale(self) -> torch.Tensor:
        """The standard deviation of the normal distribution."""
        return self.base_dist.stddev

    @property
    def mean(self) -> torch.Tensor:
        """The mean of the tanh normal distribution."""
        return SafeTanhTransformer()(self.base_dist.mean)

    @property
    def stddev(self) -> torch.Tensor:
        """The standard deviation of the tanh normal distribution."""
        return self.base_dist.stddev

    def entropy(self) -> torch.Tensor:
        """The entropy of the tanh normal distribution."""
        return self.base_dist.entropy()

    @property
    def variance(self) -> torch.Tensor:
        """The variance of the tanh normal distribution."""
        return self.base_dist.variance
