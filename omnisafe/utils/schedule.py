# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""helper class to generate scheduling params"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union


def _linear_interpolation(l, r, alpha):  # pylint: disable=invalid-name
    return l + alpha * (r - l)


class Schedule(ABC):
    """Schedule for a value based on the step"""

    @abstractmethod
    def value(self, time: Union[int, float]) -> Union[int, float]:
        """Value at time t.

        Args:
            t (float): Time.

        Returns:
            float: Value at time t.
        """


# pylint: disable=too-few-public-methods
class PiecewiseSchedule(Schedule):
    """Piece-wise schedule for a value based on the step"""

    def __init__(
        self,
        endpoints: List[Tuple[int, float]],
        outside_value=Optional[Union[int, float]],
    ) -> None:
        """From OpenAI baselines"""
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = _linear_interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, time: Union[int, float]) -> Union[int, float]:
        """Value at time t.

        Args:
            t (float): Time.
        """
        # pylint: disable=invalid-name
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= time < r_t:
                alpha = float(time - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class ConstantSchedule(Schedule):
    """Constant schedule for a value"""

    def __init__(self, value):
        """Value remains constant over time."""
        self._v = value

    def value(
        self, time: Union[int, float]
    ) -> Union[int, float]:  # pylint: disable=unused-argument
        """See Schedule.value"""
        return self._v
