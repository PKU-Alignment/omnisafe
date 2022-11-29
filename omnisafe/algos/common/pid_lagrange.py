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

import abc
from collections import deque


class PID_Lagrangian(abc.ABC):
    """Abstract base class for Lagrangian-base Algorithms."""

    def __init__(
        self,
        pid_Kp: float,
        pid_Ki: float,
        pid_Kd: float,
        pid_d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: 0.001,
        cost_limit: int,
    ):
        """init"""
        self.pid_Kp = pid_Kp
        self.pid_Ki = pid_Ki
        self.pid_Kd = pid_Kd
        self.pid_d_delay = pid_d_delay
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self.penalty_max = penalty_max
        self.sum_norm = sum_norm
        self.diff_norm = diff_norm
        self.penalty_max = lagrangian_multiplier_init
        self.pid_i = lagrangian_multiplier_init
        self.cost_ds = deque(maxlen=self.pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0
        self.cost_limit = cost_limit

    def pid_update(self, ep_cost_avg):
        """pid_update"""
        delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0.0, self.pid_i + delta * self.pid_Ki)
        if self.diff_norm:
            self.pid_i = max(0.0, min(1.0, self.pid_i))
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self.cost_ds[0])
        pid_o = self.pid_Kp * self._delta_p + self.pid_i + self.pid_Kd * pid_d
        self.cost_penalty = max(0.0, pid_o)
        if self.diff_norm:
            self.cost_penalty = min(1.0, self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self.cost_penalty = min(self.cost_penalty, self.penalty_max)
        self.cost_ds.append(self._cost_d)
