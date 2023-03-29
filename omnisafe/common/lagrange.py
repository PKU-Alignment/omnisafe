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
"""Implementation of Lagrange."""

import torch


class Lagrange:
    r"""Base class for Lagrangian-base Algorithms.

    This class implements the Lagrange multiplier update and the Lagrange loss.

    ..  note::

        Any traditional policy gradient algorithm can be converted to a Lagrangian-based algorithm
        by inheriting from this class and implementing the :meth:`_loss_pi` method.

    Example:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> def loss_pi(self, data):
        >>>     # implement your own loss function here
        >>>     return loss

    You can also inherit this class to implement your own Lagrangian-based algorithm,
    with any policy gradient method you like in ``omnisafe``.

    Example:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> class CustomAlgo:
        >>>     def __init(self) -> None:
        >>>         # initialize your own algorithm here
        >>>         super().__init__()
        >>>         # initialize the Lagrange multiplier
        >>>         self.lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound=None,
    ) -> None:
        """Initialize Lagrange multiplier."""
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        r"""Penalty loss for Lagrange multiplier.

        .. note::

            ``mean_ep_cost`` obtained from: ``self.logger.get_stats('EpCosts')[0]``, which
            are already averaged across MPI processes.

        Args:
            mean_ep_cost (float): mean episode cost.
        """
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        r"""Update Lagrange multiplier (lambda).

        Detailedly speaking, we update the Lagrange multiplier by minimizing the
        penalty loss, which is defined as:

        .. math::
            \lambda ^{'} = \lambda + \eta * (J_c - J_c^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the
        learning rate, :math:`J_c` is the mean episode cost, and :math:`J_c^*` is
        the cost limit.

        Args:
            Jc (float): mean episode cost.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]
