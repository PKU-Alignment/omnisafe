import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Unsqueeze
)



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, 3, self.heads, -1, h, w)
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)
        q, k, v = torch.flatten(qkv, start_dim=4, end_dim=5)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        out = out.view(b, self.heads, -1,  h, w)
        out = torch.flatten(out, start_dim=1, end_dim=2)

        # out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        qkv = qkv.view(b, 3, self.heads, -1, h, w)
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)
        q, k, v = torch.flatten(qkv, start_dim=4, end_dim=5)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.view(b, self.heads, -1,  h, w)
        out = torch.flatten(out, start_dim=1, end_dim=2)
        # out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Unsqueeze(),
            # Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
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

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon,
                                      kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        if self.calc_energy:
            x_inp = x

        x = x.permute(0, 2, 1)
        # x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
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
        else:
            return x

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        # x = einops.rearrange(x, 'b h t -> b t h')
        x = x.permute(0, 2, 1)
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
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
        x = x.permute(0, 2, 1)

        return x


class MLPnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        horizon=1,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
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
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

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

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + transition_dim, 1024),
            act_fn,
            nn.Linear(1024, 1024),
            act_fn,
            nn.Linear(1024, self.action_dim),
        )

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        inp = torch.cat([t, cond, x], dim=-1)
        out = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out


class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:
            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        # x = einops.rearrange(x, 'b h t -> b t h')
        x = x.permute(0, 2, 1)
        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
