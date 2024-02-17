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

# ruff: noqa
# type: ignore
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, too-many-arguments


import numpy as np
import torch
from torch import nn

from .helpers import Losses, apply_state_conditioning, cosine_beta_schedule, extract
from .temporal_unet import TemporalUnet
from .types import StateCond


class GaussianInvDynDiffusion(nn.Module):
    """
    A module that implements Gaussian Inverse Dynamics Diffusion.

    Args:
        model (TemporalUnet): The temporal U-Net model.
        horizon (int): The diffusion horizon.
        observation_dim (int): The dimension of the observation.
        action_dim (int): The dimension of the action.
        n_timesteps (int, optional): The number of diffusion timesteps. Defaults to 1000.
        clip_denoised (bool, optional): Whether to clip the denoised output. Defaults to False.
        predict_epsilon (bool, optional): Whether to predict epsilon. Defaults to True.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 256.
        loss_discount (float, optional): The discount factor for the loss. Defaults to 1.0.
        loss_weights (None, optional): The loss weights. Defaults to None.
        perform_cls_free_condition (bool, optional): Whether to perform class-free conditioning. Defaults to True.
        cls_free_condition_guidance_w (float, optional): The guidance weight for class-free conditioning. Defaults to 0.1.
    """
    def __init__(
        self,
        model: TemporalUnet,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: Optional[None] = None,
        perform_cls_free_condition: bool = True,
        cls_free_condition_guidance_w: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.model: TemporalUnet = model
        self.inv_model = ARInvModel(
            hidden_dim=hidden_dim,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
        self.perform_cls_free_condition = perform_cls_free_condition
        self.cls_free_condition_guidance_w = cls_free_condition_guidance_w

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
        """
        Sets loss coefficients for trajectory.

        Args:
            discount (float): The discount factor.

        Returns:
            torch.Tensor: The loss weights.
        """
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

    def predict_start_from_noise(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """
        If self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly.

        Args:
            x_t (torch.Tensor): The tensor at time t.
            t (int): The time step.
            noise (torch.Tensor): The noise tensor.

        Returns:
            torch.Tensor: The predicted start tensor.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start (torch.Tensor): The starting tensor for diffusion.
            x_t (torch.Tensor): The tensor at time t.
            t (int): The time step.

        Returns:
            tuple: A tuple containing the posterior mean, variance, and log variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, state_condition: StateCond, t: int, cls_free_condition_list:List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and variance for p(x_t | x_{t-1}, x_0).

        Args:
            x (torch.Tensor): The tensor at time t.
            state_condition (StateCond): The state condition.
            t (int): The time step.
            cls_free_condition_list: The class-free condition list.

        Returns:
            tuple: A tuple containing the model mean, posterior variance, and posterior log variance.
        """
        if self.perform_cls_free_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_uncond = self.model(
                x,
                t,
                cls_free_condition_list[0],
                force_dropout=True,
            )
            epsilon = epsilon_uncond
            for cls_free_condition in cls_free_condition_list:
                epsilon_cond = self.model(
                    x,
                    t,
                    cls_free_condition,
                    use_dropout=False,
                )
                epsilon += self.cls_free_condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, state_condition, t)

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
    def p_sample(self, x, state_condition, t, cls_free_condition_list):
        """
        Samples from p(x_t | x_{t-1}, x_0).

        Args:
            x (torch.Tensor): The tensor at time t.
            state_condition (StateCond): The state condition.
            t (int): The time step.
            cls_free_condition_list: The class-free condition list.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            state_condition=state_condition,
            t=t,
            cls_free_condition_list=cls_free_condition_list,
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        state_condition,
        cls_free_condition_list,
        return_diffusion=False,
    ):
        """
        Performs the sampling loop for p(x_t | x_{t-1}, x_0).

        Args:
            shape (tuple): The shape of the output tensor.
            state_condition (StateCond): The state condition.
            cls_free_condition_list: The class-free condition list.
            return_diffusion (bool, optional): Whether to return the diffusion. Defaults to False.

        Returns:
            torch.Tensor or tuple: The sampled tensor or the sampled tensor and the diffusion.
        """
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_state_conditioning(x, state_condition, 0)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, state_condition, timesteps, cls_free_condition_list)
            x = apply_state_conditioning(x, state_condition, 0)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(
        self,
        state_condition,
        cls_free_condition_list,
        *args,
        horizon=None,
        **kwargs,
    ):
        """
        Generates a conditional sample.

        Args:
            state_condition (StateCond): The state condition.
            cls_free_condition_list: The class-free condition list.
            horizon (int, optional): The diffusion horizon. Defaults to None.

        Returns:
            torch.Tensor: The generated sample.
        """
        batch_size = len(state_condition[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, state_condition, cls_free_condition_list, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        Samples from q(x_t | x_{t-1}).

        Args:
            x_start (torch.Tensor): The starting tensor for diffusion.
            t (int): The time step.
            noise (torch.Tensor, optional): The noise tensor. Defaults to None.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(
        self, x_start: torch.Tensor, state_condition: StateCond, t: int, cls_free_condition=None
    ):
        """
        Calculates the loss and additional information for the diffusion model.

        Args:
            x_start (torch.Tensor): The starting tensor for diffusion.
            state_condition (StateCond): The state condition for diffusion.
            t (int): The time step for diffusion.
            cls_free_condition (optional): The class-free condition for diffusion.

        Returns:
            tuple: A tuple containing the loss and additional information.
        """
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_state_conditioning(x_noisy, state_condition, 0)

        x_recon = self.model(x_noisy, state_condition, t, cls_free_condition)

        if not self.predict_epsilon:
            x_recon = apply_state_conditioning(x_recon, state_condition, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x: torch.Tensor, state_condition: StateCond, cls_free_condition=None):
        """
        Calculates the loss function for the diffusion model.

        Args:
            x (torch.Tensor): The input tensor.
            state_condition (StateCond): The state condition.
            cls_free_condition (optional): The class-free condition.

        Returns:
            torch.Tensor: The loss value.
        """
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(
            x[:, :, self.action_dim :],
            state_condition,
            t,
            cls_free_condition,
        )
        # Calculating inv loss
        x_t = x[:, :-1, self.action_dim :]
        a_t = x[:, :-1, : self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        a_t = a_t.reshape(-1, self.action_dim)
        inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)

        loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info

    def forward(self, state_condition, *args, **kwargs):
        """
        Performs the forward pass of the diffusion model.

        Args:
            state_condition: The state condition for the diffusion model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the conditional sample operation.
        """
        return self.conditional_sample(state_condition, *args, **kwargs)


class ARInvModel(nn.Module):
    """
    The ARInvModel class implements the auto regressive inverse model.
    """

    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        """
        Initializes the ARInvModel class.

        Args:
            hidden_dim (int): The dimension of the hidden layer.
            observation_dim (int): The dimension of the observation.
            action_dim (int): The dimension of the action.
            low_act (float, optional): The lower bound of the action range. Defaults to -1.0.
            up_act (float, optional): The upper bound of the action range. Defaults to 1.0.
        """
        super().__init__()
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

        self.lin_mod = nn.ModuleList(
            [nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)],
        )
        self.act_mod = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                ),
            ],
        )

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                ),
            )

    def forward(self, comb_state, deterministic=False):
        """
        Forward pass of the diffusion model.

        Args:
            comb_state (torch.Tensor): Combined state input.
            deterministic (bool, optional): Flag indicating whether to use deterministic sampling.
                                            Defaults to False.

        Returns:
            torch.Tensor: Concatenated actions.
        """
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(
                self.low_act + l_0 * self.bin_size,
                self.low_act + (l_0 + 1) * self.bin_size,
            ).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](
                torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1),
            )
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(
                    self.low_act + l_i * self.bin_size,
                    self.low_act + (l_i + 1) * self.bin_size,
                ).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        """
        Calculates the loss for the diffusion model.

        Args:
            comb_state (torch.Tensor): The combined state input.
            action (torch.Tensor): The action input.

        Returns:
            torch.Tensor: The calculated loss.
        """
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(
                self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                l_action[:, i],
            )

        return loss / self.action_dim
