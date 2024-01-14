# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""helper class to generate scheduling params."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


def _linear_interpolation(left: float, right: float, alpha: float) -> float:
    return left + alpha * (right - left)


class Schedule(ABC):
    """Schedule for a value based on the step."""

    @abstractmethod
    def value(self, time: float) -> float:
        """Value at time t."""


# pylint: disable=too-few-public-methods
class PiecewiseSchedule(Schedule):
    """Piece-wise schedule for a value based on the step, from OpenAI baselines.

    Args:
        endpoints (list[tuple[int, float]]): List of pairs `(time, value)` meaning that schedule
            will output `value` when `t==time`. All the values for time must be sorted in an
            increasing order. When t is between two times, e.g. `(time_a, value_a)` and
            `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs is interpolated
            linearly between `value_a` and `value_b`.
        outside_value (int or float): Value to use if `t` is before the first time in `endpoints` or
            after the last one.
    """

    def __init__(
        self,
        endpoints: list[tuple[int, float]],
        outside_value: float,
    ) -> None:
        """Initialize an instance of :class:`PiecewiseSchedule`."""
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation: Callable[[float, float, float], float] = _linear_interpolation
        self._outside_value: float = outside_value
        self._endpoints: list[tuple[int, float]] = endpoints

    def value(self, time: float) -> float:
        """Value at time t.

        Args:
            time (int or float): Current time step.

        Returns:
            The interpolation value at time t or outside_value if t is before the first time in
            endpoints of after the last one.

        Raises:
            AssertionError: If the time is not in the endpoints.
        """
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= time < right_t:
                alpha = float(time - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class ConstantSchedule(Schedule):
    """Constant schedule for a value."""

    def __init__(self, value: float) -> None:
        """Initialize an instance of :class:`ConstantSchedule`."""
        self._v: float = value

    def value(self, time: float) -> float:  # pylint: disable=unused-argument
        """Value at time t.

        Args:
            time (int or float): Current time step.

        Returns:
            The interpolation value at time t or outside_value if t is before the first time in
            endpoints of after the last one.
        """
        return self._v
