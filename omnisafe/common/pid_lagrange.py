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
"""Implementation of PID Lagrange."""

from __future__ import annotations

import abc
from collections import deque
from typing import Deque


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class PIDLagrangian(abc.ABC):  # noqa: B024
    """Abstract base class for Lagrangian-base Algorithms.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient algorithm.
        As PID_Lagrange use the PID controller to control the lagrangian multiplier,
        it is more stable than the naive Lagrange.

    References:

    - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
    - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
    - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        pid_kp: float,
        pid_ki: float,
        pid_kd: float,
        pid_d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: int,
    ) -> None:
        """Initialize PIDLagrangian.

        Args:
            pid_kp: The proportional gain of the PID controller.
            pid_ki: The integral gain of the PID controller.
            pid_kd: The derivative gain of the PID controller.
            pid_d_delay: The delay of the derivative term of the PID controller.
            pid_delta_p_ema_alpha: The exponential moving average alpha of the proportional term of the PID controller.
            pid_delta_d_ema_alpha: The exponential moving average alpha of the derivative term of the PID controller.
            sum_norm: Whether to normalize the sum of the cost.
            diff_norm: Whether to normalize the difference of the cost.
            penalty_max: The maximum penalty.
            lagrangian_multiplier_init: The initial value of the lagrangian multiplier.
            cost_limit: The cost limit.
        """
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pid_d_delay = pid_d_delay
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self.penalty_max = penalty_max
        self.sum_norm = sum_norm
        self.diff_norm = diff_norm
        self.pid_i = lagrangian_multiplier_init
        self.cost_ds: Deque[float] = deque(maxlen=self.pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p: float = 0
        self._cost_d: float = 0
        self.cost_limit: float = cost_limit
        self.cost_penalty: float = 0

    def pid_update(self, ep_cost_avg: float) -> None:
        r"""Update the PID controller.

        Detailedly, PID controller update the lagrangian multiplier following the next equation:

        .. math::
            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        delta = float(ep_cost_avg - self.cost_limit)
        self.pid_i = max(0.0, self.pid_i + delta * self.pid_ki)
        if self.diff_norm:
            self.pid_i = max(0.0, min(1.0, self.pid_i))
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self.cost_ds[0])
        pid_o = self.pid_kp * self._delta_p + self.pid_i + self.pid_kd * pid_d
        self.cost_penalty = max(0.0, pid_o)
        if self.diff_norm:
            self.cost_penalty = min(1.0, self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self.cost_penalty = min(self.cost_penalty, self.penalty_max)
        self.cost_ds.append(self._cost_d)
