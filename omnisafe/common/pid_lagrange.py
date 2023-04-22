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
        self._pid_kp: float
        self._pid_ki: float
        self._pid_kd: float
        self._pid_d_delay: int
        self._pid_delta_p_ema_alpha: float
        self._pid_delta_d_ema_alpha: float
        self._penalty_max: int
        self._sum_norm: bool
        self._diff_norm: bool
        self._pid_i: float
        self._cost_ds: Deque[float]
        self._delta_p: float
        self._cost_d: float
        self._cost_limit: float
        self._cost_penalty: float

        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self._penalty_max = penalty_max
        self._sum_norm = sum_norm
        self._diff_norm = diff_norm
        self._pid_i = lagrangian_multiplier_init
        self._cost_ds = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0
        self._cost_limit = cost_limit
        self._cost_penalty = 0

    @property
    def lagrangian_multiplier(self) -> float:
        """Return the current value of the lagrangian multiplier."""
        return self._cost_penalty

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
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
