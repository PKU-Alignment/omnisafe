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
"""Helpful functions for diffusion model."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class Unsqueeze(nn.Module):
    """Implementation of Unsqueeze module."""

    def __init__(self, dim: int = -1) -> None:
        """Initialize for Class:Unsqueeze."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of Unsqueeze."""
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    """Implementation of squeeze module."""

    def __init__(self, dim: int = -1) -> None:
        """Initialize for Class:squeeze."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of squeeze."""
        return x.squeeze(self.dim)


class SinusoidalPosEmb(nn.Module):
    """Implementation of SinusoidalPosEmb module."""

    def __init__(self, dim: int) -> None:
        """Initialize for Class:SinusoidalPosEmb."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of SinusoidalPosEmb."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    """Implementation of Downsample1d module."""

    def __init__(self, dim: int) -> None:
        """Initialize for Class:Downsample1d."""
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of Downsample1d."""
        return self.conv(x)


class Upsample1d(nn.Module):
    """Implementation of Upsample1d module."""

    def __init__(self, dim: int) -> None:
        """Initialize for Class:Upsample1d."""
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of Upsample1d."""
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Implementation of Conv1dBlock module.

    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        mish: bool = True,
        n_groups: int = 8,
    ) -> None:
        """Initialize for Class:Conv1dBlock."""
        super().__init__()

        act_fn = nn.Mish() if mish else nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Unsqueeze(2),
            nn.GroupNorm(n_groups, out_channels),
            Squeeze(2),
            act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of Unsqueeze."""
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor) -> torch.Tensor:
    """Calculate for extracting t step value."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: object = torch.float32,
) -> torch.Tensor:
    """Calculate cosine beta schedule value.

    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


# def apply_conditioning(x, conditions, action_dim):
#     for t, val in conditions.items():
#         x[:, :t, action_dim:] = val.clone()
#     # x[:,0:,action_dim]=conditions.clone()
#     return x


def history_cover(
    x: torch.Tensor,
    history: torch.Tensor,
    action_dim: int,
    history_length: int,
) -> torch.Tensor:
    """Update observation history queue."""
    history_length = history.shape[-2]
    x[:, :history_length, action_dim:] = history.clone()
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    """Implementation of WeightedLoss module."""

    def __init__(self, weights: float, action_dim: int) -> None:
        """Initialize for Class:WeightedLoss."""
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Forward progress of WeightedLoss."""
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class WeightedStateLoss(nn.Module):
    """Implementation of WeightedStateLoss module."""

    def __init__(self, weights: float) -> None:
        """Initialize for Class:WeightedStateLoss."""
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> tuple:
        """Forward progress of WeightedStateLoss.

        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}


class ValueLoss(nn.Module):
    """Implementation of ValueLoss module."""

    def __init__(self, *args: tuple) -> None:
        """Initialize for Class:ValueLoss."""
        super().__init__()

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> tuple:
        """Forward progress of ValueLoss."""
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                pred.detach().cpu().numpy().squeeze(),
                targ.detach().cpu().numpy().squeeze(),
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(),
            'mean_targ': targ.mean(),
            'min_pred': pred.min(),
            'min_targ': targ.min(),
            'max_pred': pred.max(),
            'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):
    """Implementation of WeightedL1 module."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Loss function of WeightedL1."""
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    """Implementation of WeightedL2 module."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Loss function of WeightedL2."""
        return F.mse_loss(pred, targ, reduction='none')


class WeightedStateL2(WeightedStateLoss):
    """Implementation of WeightedStateL2 module."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Loss function of WeightedStateL2."""
        return F.mse_loss(pred, targ, reduction='none')


class ValueL1(ValueLoss):
    """Implementation of ValueL1 module."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Loss function of ValueL1."""
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    """Implementation of ValueL2 module."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Loss function of ValueL2."""
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}
