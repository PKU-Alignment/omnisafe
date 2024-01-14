import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.distributions import Bernoulli
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
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
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
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
            Rearrange('batch t -> batch t 1'),
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
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
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

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
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

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
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

        x = einops.rearrange(x, 'b t h -> b h t')

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
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
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
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        inp = torch.cat([t, cond, x], dim=-1)
        out  = self.mlp(inp)

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

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
                 condition_guidance_w=0.1, ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            noise[:, 0, self.action_dim:] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
                 condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_dataset_normalizer(self, normalizer):
        self.dataset_normalizer = normalizer

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):

        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'loss_inv': loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'loss_inv': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)
            loss = (1 / 2) * (diffuse_loss + inv_loss)
            info['loss_diffuser'] = diffuse_loss
            info['loss_inv'] = inv_loss
            info['loss_total'] = loss

        return loss, info

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action from observation.

        deterministic is not used in this method, it is just for compatibility.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to return deterministic action. Defaults to False.

        Returns:
            torch.Tensor: Action.
        """
        device = obs.device
        obs = torch.from_numpy(self.dataset_normalizer.normalize(obs.cpu().numpy(), 'observations')).to(device)
        returns = 0.9 * torch.ones(1, 1, device=device)
        conditions = {0: obs}
        samples = self.conditional_sample(conditions, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * self.observation_dim)
        action = self.inv_model(obs_comb)
        action = torch.from_numpy(self.dataset_normalizer.unnormalize(action.detach().cpu().numpy(), 'actions')).to(device)
        return action

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                 l_action[:, i])

        return loss / self.action_dim


class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
                 condition_guidance_w=0.1, ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        if self.model.calc_energy:
            assert self.predict_epsilon
            action_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        pred = self.model(action_noisy, state, t, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {'a0_loss': loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1  # Assumes horizon=1
        x = x[:, 0, :]
        cond = x[:, self.action_dim:]  # Observation
        x = x[:, :self.action_dim]  # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
