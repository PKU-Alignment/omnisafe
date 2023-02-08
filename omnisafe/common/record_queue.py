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
"""Implementation of Record Queue."""

from collections import deque

import numpy as np

from omnisafe.typing import List


class RecordQueue:
    """RecordQueue."""

    def __init__(self, *names, maxlen=100) -> None:
        """Initialize the RecordQueue."""
        self.queues = {}
        self._create_deques(*names, maxlen=maxlen)

    def _create_deques(self, *names, maxlen=100) -> None:
        """Create queues by names."""
        for name in names:
            self.queues[name] = deque(maxlen=maxlen)

    def append(self, **kwargs) -> None:
        """Add values to the queues."""
        for key, value in kwargs.items():
            assert key in self.queues, f'{key} has not been set in queues {self.queues.keys()}'
            self.queues[key].append(value)

    def non_empty_mean(self, name) -> np.ndarray:
        """Get the mean of the non-empty values."""
        return np.mean(self.queues[name]) if len(self.queues[name]) else 0.0

    def get_mean(self, *names) -> List:
        """Get the means of needed queue names."""
        assert all(
            name in self.queues for name in names
        ), f'{names} has not been set in queues {self.queues.keys()}'
        if len(names) == 1:
            return self.non_empty_mean(names[0])
        return [self.non_empty_mean(name) for name in names]

    def reset(self, *names) -> None:
        """Reset the needed queue."""
        assert all(
            name in self.queues for name in names
        ), f'{names} has not been set in queues {self.queues.keys()}'
        for name in names:
            self.queues[name].clear()
