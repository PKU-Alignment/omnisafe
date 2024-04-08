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
"""Implementation of diffusion model."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from omnisafe.models.dd_models.helpers import Losses, cosine_beta_schedule, extract, history_cover


# pylint: disable=R,W,C
class GaussianInvDynDiffusion(nn.Module):
    """Implementation of GaussianInvDynDiffusion."""

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        clip_denoised: float = False,
        predict_epsilon: float = True,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        returns_condition: bool = False,
        condition_guidance_w: float = 0.1,
        train_only_inv: bool = False,
        history_length: int = 1,
        multi_step_pred: int = 1,
    ) -> None:
        """Initialize for Class:GaussianInvDynDiffusion."""
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.train_only_inv = train_only_inv
        self.history_lenght = history_length
        self.history_obs = []
        self.history_count = 0
        self.multi_step_pred = multi_step_pred
        self.multi_step_count = 0

        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
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
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount: float) -> torch.Tensor:
        """Get loss weights for training model."""
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """If self.predict_epsilon, model output is (scaled) noise, otherwise, model predicts x0 directly."""
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )

        return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple:
        """Calculate q-sample posterior.

        Args:
            x_start: step 0
            x_t: step t
            t: diffusion step

        Returns:
            q_m,q_v
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
    ) -> tuple:
        """Calculate for p-sample mean and variance.

        Args:
            constraints: constraints for condition generate
            skills: skills for condition generate

        Returns:
            p_m,p_v
        """
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, returns, constraints, skills, use_dropout=False)
            epsilon_uncond = self.model(x, t, returns, constraints, skills, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, t)
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
    ) -> torch.Tensor:
        """The process of single p-sample.

        Returns:
            p_sample noise

        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            returns=returns,
            constraints=constraints,
            skills=skills,
        )
        noise = 0.5 * torch.randn_like(x, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: torch.Tensor,
        history: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
        verbose: torch.Tensor = True,
        return_diffusion: object = False,
    ) -> object:
        """The process of p-sample loop.

        Returns:
            p_sample_loop noise
        """
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = history_cover(x, history, 0, self.history_lenght)

        if return_diffusion:
            diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, returns, constraints, skills)
            x = history_cover(x, history, 0, self.history_lenght)

            # progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)

        return x

    @torch.no_grad()
    def conditional_sample(
        self,
        obs_history: torch.Tensor,
        returns: torch.Tensor = None,
        horizon: torch.Tensor = None,
        *args: tuple,
        **kwargs: dict,
    ) -> torch.Tensor:
        """The process of conditional sample.

        Returns:
            p_sample value

        """
        batch_size = len(obs_history)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, obs_history, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """The process of q-sample.

        Returns:
            q_sample

        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate p-sample loss value."""
        history = x_start[:, : self.history_lenght, :]
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = history_cover(x_noisy, history, 0, self.history_lenght)
        x_recon = self.model(x_noisy, t, returns, constraints=constraints, skills=skills)

        if not self.predict_epsilon:
            x_recon = history_cover(x_recon, history, 0, self.history_lenght)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(
        self,
        x: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
    ) -> torch.Tensor:
        """The process of loss training."""
        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim :]
            a_t = x[:, :-1, : self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim :]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)

            pred_a_t = self.inv_model(x_comb_t)
            loss = F.mse_loss(pred_a_t, a_t)
            info = {'loss_inv': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(
                x[:, :, self.action_dim :],
                t,
                returns,
                constraints,
                skills,
            )
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim :]
            a_t = x[:, :-1, : self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim :]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)

            pred_a_t = self.inv_model(x_comb_t)
            inv_loss = F.mse_loss(pred_a_t, a_t)
            loss = (1 / 2) * (diffuse_loss + inv_loss)
            info['loss_diffuser'] = diffuse_loss
            info['loss_inv'] = inv_loss
            info['loss_total'] = loss

        return loss, info

    def history_obs_update(self, obs: torch.Tensor) -> torch.Tensor:
        """Maintain the history observation queue."""
        if self.history_count % self.n_timesteps == 0:
            self.history_obs = [torch.zeros((1, self.observation_dim))] * self.history_lenght
        self.history_obs.append(obs)
        self.history_obs.pop(0)
        history_obs = torch.stack(self.history_obs, dim=1)
        self.history_count += 1
        return history_obs

    def forward(self, *args: tuple, **kwargs: dict) -> torch.Tensor:
        """Diffusion model forward function."""
        return self.conditional_sample(*args, **kwargs)
