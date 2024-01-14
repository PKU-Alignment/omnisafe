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
"""Implementation of Normalizer."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Calculate normalized raw_data from running mean and std.

    References:
        - Title: Updating Formulae and a Pairwise Algorithm for Computing Sample Variances
        - Author: Tony F. Chan, Gene H. Golub, Randall J. LeVeque
        - URL: `Normalizer <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`_
    """

    _mean: torch.Tensor  # running mean
    _sumsq: torch.Tensor  # running sum of squares
    _var: torch.Tensor  # running variance
    _std: torch.Tensor  # running standard deviation
    _count: torch.Tensor  # number of samples
    _clip: torch.Tensor  # clip value

    def __init__(self, shape: tuple[int, ...], clip: float = 1e6) -> None:
        """Initialize an instance of :class:`Normalizer`."""
        super().__init__()
        if shape == ():
            self.register_buffer('_mean', torch.tensor(0.0))
            self.register_buffer('_sumsq', torch.tensor(0.0))
            self.register_buffer('_var', torch.tensor(0.0))
            self.register_buffer('_std', torch.tensor(0.0))
            self.register_buffer('_count', torch.tensor(0))
            self.register_buffer('_clip', clip * torch.tensor(1.0))
        else:
            self.register_buffer('_mean', torch.zeros(*shape))
            self.register_buffer('_sumsq', torch.zeros(*shape))
            self.register_buffer('_var', torch.zeros(*shape))
            self.register_buffer('_std', torch.zeros(*shape))
            self.register_buffer('_count', torch.tensor(0))
            self.register_buffer('_clip', clip * torch.ones(*shape))

        self._shape: tuple[int, ...] = shape
        self._first: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the normalize."""
        return self._shape

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the normalize."""
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        """Return the std of the normalize."""
        return self._std

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data.

        Args:
            data (torch.Tensor): The raw data to be normalized.

        Returns:
            The normalized data.
        """
        return self.normalize(data)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data.

        .. hint::
            - If the data is the first data, the data will be used to initialize the mean and std.
            - If the data is not the first data, the data will be normalized by the mean and std.
            - Update the mean and std by the data.

        Args:
            data (torch.Tensor): The raw data to be normalized.

        Returns:
            The normalized data.
        """
        data = data.to(self._mean.device)
        self._push(data)
        if self._count <= 1:
            return data
        output = (data - self._mean) / self._std
        return torch.clamp(output, -self._clip, self._clip)

    def _push(self, raw_data: torch.Tensor) -> None:
        """Update the mean and std by the raw_data.

        Args:
            raw_data (torch.Tensor): The raw data to be normalized.
        """
        if raw_data.shape == self._shape:
            raw_data = raw_data.unsqueeze(0)
        assert raw_data.shape[1:] == self._shape, 'data shape must be equal to (batch_size, *shape)'

        if self._first:
            self._mean = torch.mean(raw_data, dim=0)
            self._sumsq = torch.sum((raw_data - self._mean) ** 2, dim=0)
            self._count = torch.tensor(
                raw_data.shape[0],
                dtype=self._count.dtype,
                device=self._count.device,
            )
            self._first = False
        else:
            count_raw = raw_data.shape[0]
            count = self._count + count_raw
            mean_raw = torch.mean(raw_data, dim=0)
            delta = mean_raw - self._mean
            self._mean += delta * count_raw / count
            sumq_raw = torch.sum((raw_data - mean_raw) ** 2, dim=0)
            self._sumsq += sumq_raw + delta**2 * self._count * count_raw / count
            self._count = count
        self._var = self._sumsq / (self._count - 1)
        self._std = torch.sqrt(self._var)
        self._std = torch.max(self._std, 1e-2 * torch.ones_like(self._std))

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load the state_dict to the normalizer.

        Args:
            state_dict (Mapping[str, Any]): The state_dict to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys in :attr:`state_dict`.
                Defaults to True.

        Returns:
            The loaded normalizer.
        """
        self._first = False
        return super().load_state_dict(state_dict, strict, assign)
