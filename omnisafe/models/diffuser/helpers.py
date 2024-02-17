# Copyright 2022-2024 OmniSafe Team. All Rights Reserved.
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
"""Diffuser helpers."""

# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim: int) -> None:
        """Initialize the sinusoidal positional embedding."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the sinusoidal positional embedding."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    """Downsample the input by a factor of 2."""

    def __init__(self, dim: int) -> None:
        """Initialize the Downsample1d."""
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the down."""
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsample the input by a factor of 2."""

    def __init__(self, dim: int) -> None:
        """Initialize the Upsample1d."""
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the up."""
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        mish: bool = True,
        n_groups: int = 8,
    ) -> None:
        """Initialize the Conv1dBlock."""
        super().__init__()

        act_fn: nn.Module = nn.Mish() if mish else nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Conv."""
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract the values from a tensor using the indices in t."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ for beta."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_state_conditioning(
    x: torch.Tensor,
    state_conditions: Dict[int, torch.Tensor],
    action_dim: int = 0,
) -> torch.Tensor:
    """Apply the state conditioning to the input tensor."""
    # action dim is always 0 for decision diffusion
    # for compatibility reason, see plan diffuser
    for t, val in state_conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedStateL2(nn.Module):
    """Weighted L2 loss."""

    def __init__(self, weights: torch.Tensor) -> None:
        """Initialize the weighted L2 loss."""
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted L2 loss between the predicted and target values."""
        loss = self._loss(pred, target)
        return (loss * self.weights).mean()

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, targ, reduction='none')
