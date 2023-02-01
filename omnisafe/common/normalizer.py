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
"""Implementation of Vector Buffer."""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Calculate normalized raw_data from running mean and std

    See https://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, shape, clip=1e6):
        """Initialize the normalize."""
        super().__init__()
        self.raw_data = nn.Parameter(
            torch.zeros(*shape), requires_grad=False
        )  # Current value of data stream
        self.mean = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current mean
        self.sumsq = nn.Parameter(
            torch.zeros(*shape), requires_grad=False
        )  # Current sum of squares, used in var/std calculation

        self.var = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current variance
        self.std = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current std

        self.count = nn.Parameter(torch.zeros(1), requires_grad=False)  # Counter

        self.clip = nn.Parameter(clip * torch.ones(*shape), requires_grad=False)

    def push(self, raw_data):
        """Push a new value into the stream."""
        self.raw_data.data = raw_data
        self.count.data[0] += 1
        if self.count.data[0] == 1:
            self.mean.data = raw_data
        else:
            old_mean = self.mean
            self.mean.data += (raw_data - self.mean.data) / self.count.data
            self.sumsq.data += (raw_data - old_mean.data) * (raw_data - self.mean.data)
            self.var.data = self.sumsq.data / (self.count.data - 1)
            self.std.data = torch.sqrt(self.var.data)
            self.std.data = torch.max(self.std.data, 1e-2 * torch.ones_like(self.std.data))

    def forwarad(self, raw_data=None):
        """Normalize the raw_data."""
        return self.normalize(raw_data)

    def pre_process(self, raw_data):
        """Pre-process the raw_data."""
        if len(raw_data.shape) == 1:
            raw_data = raw_data.unsqueeze(-1)
        return raw_data

    def normalize(self, raw_data=None):
        """Normalize the raw_data."""
        raw_data = self.pre_process(raw_data)
        self.push(raw_data)
        if self.count <= 1:
            return self.raw_data.data
        output = (self.raw_data.data - self.mean.data) / self.std.data
        return torch.clamp(output, -self.clip.data, self.clip.data)
