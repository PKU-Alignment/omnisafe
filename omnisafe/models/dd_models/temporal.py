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
"""Implementation of UTemporalModel."""

import types

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from omnisafe.models.dd_models.helpers import (
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Unsqueeze,
    Upsample1d,
)


class Residual(nn.Module):
    """Implementation of Residual Module."""

    def __init__(self, fn: types.FunctionType) -> None:
        """Initialize for Class:Residual."""
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args: tuple, **kwargs: dict) -> torch.Tensor:
        """Forward progress of Residual."""
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """Implementation of PreNorm Module."""

    def __init__(self, dim: int, fn: types.FunctionType) -> None:
        """Initialize for Class:PreNorm."""
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of PreNorm."""
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    """Implementation of LinearAttention Module."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 128) -> None:
        """Initialize for Class:LinearAttention."""
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of LinearAttention."""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, 3, self.heads, -1, h, w)
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)
        q, k, v = torch.flatten(qkv, start_dim=4, end_dim=5)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        out = out.view(b, self.heads, -1, h, w)
        out = torch.flatten(out, start_dim=1, end_dim=2)

        # out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    """Implementation of GlobalMixing Module."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 128) -> None:
        """Initialize for Class:GlobalMixing."""
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward progress of GlobalMixing."""
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        qkv = qkv.view(b, 3, self.heads, -1, h, w)
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)
        q, k, v = torch.flatten(qkv, start_dim=4, end_dim=5)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.view(b, self.heads, -1, h, w)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        # out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    """Implementation of ResidualTemporalBlock Module."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        horizon: int,
        kernel_size: int = 5,
        mish: bool = True,
    ) -> None:
        """Initialize for Class:ResidualTemporalBlock."""
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
            Unsqueeze(),
            # Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward progress of ResidualTemporalBlock.

        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]

        Returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """Implementation of TemporalUnet."""

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        dim_mults: tuple = (1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        calc_energy: bool = False,
        kernel_size: int = 5,
        constraints_dim: int = 0,
        skills_dim: int = 0,
    ) -> None:
        """Initialize for Class:TemporalUnet."""
        super().__init__()

        dims = [transition_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
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

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.constraints_dim = constraints_dim
        self.skills_dim = skills_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        if self.constraints_dim:
            self.constraints_mlp = nn.Sequential(
                nn.Linear(constraints_dim, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim += dim

        if self.skills_dim:
            self.skills_mlp = nn.Sequential(
                nn.Linear(skills_dim, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim += dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            horizon=horizon,
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
            horizon=horizon,
            kernel_size=kernel_size,
            mish=mish,
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            horizon=horizon,
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
                            horizon=horizon,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=embed_dim,
                            horizon=horizon,
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
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ) -> torch.Tensor:
        """Forward progress of TemporalUnet.

        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """
        if self.calc_energy:
            x_inp = x

        x = x.permute(0, 2, 1)
        # x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        embed_list = []
        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            embed_list.append(returns_embed)

        if self.constraints_dim:
            assert constraints is not None
            constraints_embed = self.constraints_mlp(constraints)
            embed_list.append(constraints_embed)

        if self.skills_dim:
            assert skills is not None
            skills_embed = self.skills_mlp(skills)
            embed_list.append(skills_embed)

        embed = torch.cat(embed_list, dim=-1)
        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(embed.size(0), 1)).to(embed.device)
            embed = mask * embed
        if force_dropout:
            embed = 0 * embed
        t = torch.cat([t, embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # import pdb; pdb.set_trace()

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        # x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]

        return x

    def get_pred(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor,
        returns: torch.Tensor = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ) -> torch.Tensor:
        """Predict progress of TemporalUnet.

        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """
        # x = einops.rearrange(x, 'b h t -> b t h')
        x = x.permute(0, 2, 1)
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(
                    returns_embed.device,
                )
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

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

        # x = einops.rearrange(x, 'b t h -> b h t')
        return x.permute(0, 2, 1)
