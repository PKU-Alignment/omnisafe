"""
Temporal U-Net model.

This module implements the Temporal U-Net model, which is a type of convolutional neural network
designed for processing temporal data. It consists of several components such as residual blocks,
linear attention, global mixing, and downsampling/upsampling layers.

Classes:
- Residual: Residual module.
- PreNorm: PreNorm module.
- LinearAttention: LinearAttention module.
- GlobalMixing: GlobalMixing module.
- ResidualTemporalBlock: Residual Temporal Block module.
- TemporalUnet: Temporal U-Net module.
"""

# ... (existing code)
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


# pylint: disable=too-many-instance-attributes, too-many-arguments

"""Temporal U-Net model."""

from typing import List, Optional

import einops
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from omnisafe.models.diffuser.helpers import (  # type: ignore
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)


class Residual(nn.Module):
    """
    Residual module.

    Args:
        fn (nn.Module): The function module to be applied to the input tensor.

    Attributes:
        fn (nn.Module): The function module to be applied to the input tensor.
    """

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fn(x) + x


class PreNorm(nn.Module):
    """
    PreNorm module.

    This module applies instance normalization to the input tensor before passing it through the given module.

    Args:
        dim (int): Number of channels in the input tensor.
        fn (nn.Module): The module to be applied to the normalized input tensor.

    Attributes:
        fn (nn.Module): The module to be applied to the normalized input tensor.
        norm (nn.InstanceNorm2d): Instance normalization layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the PreNorm module.

    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    """
    LinearAttention module performs linear attention mechanism.

    Args:
        dim (int): Input dimension of the tensor.
        heads (int, optional): Number of attention heads. Defaults to 4.
        dim_head (int, optional): Dimension of each attention head. Defaults to 128.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 128) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        _, _, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            'b (qkv heads c) h w -> qkv b heads c (h w)',
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    """
    GlobalMixing module performs global mixing of features using self-attention mechanism.
    Args:
        dim (int): Input feature dimension.
        heads (int, optional): Number of attention heads. Defaults to 4.
        dim_head (int, optional): Dimension of each attention head. Defaults to 128.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 128) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GlobalMixing module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        _, _, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            'b (qkv heads c) h w -> qkv b heads c (h w)',
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    """
    Residual Temporal Block module.

    Args:
        inp_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        embed_dim (int): Dimension of the embedding tensor.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
        mish (bool, optional): Whether to use Mish activation function. Defaults to True.

    Attributes:
        blocks (nn.ModuleList): List of convolutional blocks.
        time_mlp (nn.Sequential): Sequential module for temporal MLP.
        residual_conv (nn.Module): Residual convolutional layer.

    Methods:
        forward(x, t): Forward pass of the ResidualTemporalBlock module.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        mish: bool = True,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
                Conv1dBlock(out_channels, out_channels, kernel_size, mish),
            ],
        )

        act_fn = nn.Mish() if mish else nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualTemporalBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inp_channels, horizon).
            t (torch.Tensor): Input tensor of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, horizon).
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """
    Temporal U-Net module.

    Args:
        horizon (int): The length of the time horizon.
        transition_dim (int): The dimension of the transition.
        cls_free_condition_dim (int, optional): The dimension of the class-free condition. Defaults to 1.
        dim (int, optional): The base dimension. Defaults to 128.
        dim_mults (List[int], optional): The list of dimension multipliers for each resolution level. Defaults to None.
        cls_free_condition (bool, optional): Whether to use class-free condition. Defaults to True.
        condition_dropout (float, optional): The dropout rate for the condition. Defaults to 0.1.
        kernel_size (int, optional): The kernel size for the convolutional blocks. Defaults to 5.

    Attributes:
        time_dim (int): The dimension of the time input.
        returns_dim (int): The dimension of the returns input.

    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cls_free_condition_dim: int = 1,
        dim: int = 128,
        dim_mults: Optional[List[int]] = None,
        cls_free_condition: bool = True,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        dim_mults = dim_mults or [1, 2, 4, 8]
        dims = [transition_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.cls_free_condition = cls_free_condition
        self.condition_dropout = condition_dropout

        if self.cls_free_condition:
            self.cls_free_mlp = nn.Sequential(
                nn.Linear(cls_free_condition_dim, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ],
                ),
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ],
                ),
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the TemporalUnet module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, horizon, transition).
            time (torch.Tensor): Input tensor of shape (batch_size, dim).
            returns (torch.Tensor, optional): Input tensor of shape (batch_size, horizon). Defaults to None.
            use_dropout (bool, optional): Whether to use dropout. Defaults to True.
            force_dropout (bool, optional): Whether to force dropout. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, transition_dim, horizon).
        """
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.cls_free_condition:
            assert returns is not None
            cls_free_embed = self.cls_free_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(cls_free_embed.size(0), 1)).to(
                    cls_free_embed.device,
                )
                cls_free_embed = mask * cls_free_embed
            if force_dropout:
                cls_free_embed = 0 * cls_free_embed
            t = torch.cat([t, cls_free_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        return einops.rearrange(x, 'b t h -> b h t')
