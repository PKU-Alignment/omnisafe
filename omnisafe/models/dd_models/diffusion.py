import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    history_cover,
    Losses,
)


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 clip_denoised=False, predict_epsilon=True, hidden_dim=256, loss_discount=1.0, returns_condition=False,
                 condition_guidance_w=0.1, train_only_inv=False, history_length=1,
                 multi_step_pred=1, test_constraints=None, test_skills=None):
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

    def p_mean_variance(self, x, t, returns=None, constraints=None, skills=None):
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
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, returns=None, constraints=None, skills=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, returns=returns,
                                                                 constraints=constraints, skills=skills)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, history, returns=None, constraints=None, skills=None, verbose=True,
                      return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = history_cover(x, history, 0, self.history_lenght)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, returns, constraints, skills)
            x = history_cover(x, history, 0, self.history_lenght)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, obs_history, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(obs_history)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, obs_history, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, returns=None, constraints=None, skills=None):
        history = x_start[:, :self.history_lenght, :]
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

    def loss(self, x, returns=None, constraints=None, skills=None):

        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)

            pred_a_t = self.inv_model(x_comb_t)
            loss = F.mse_loss(pred_a_t, a_t)
            info = {'loss_inv': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], t, returns, constraints, skills)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
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

    def history_obs_update(self, obs):
        if self.history_count % self.n_timesteps == 0:
            self.history_obs = [torch.zeros((1, self.observation_dim))] * self.history_lenght
        self.history_obs.append(obs)
        self.history_obs.pop(0)
        history_obs = torch.stack(self.history_obs, dim=1)
        self.history_count += 1
        return history_obs

    # def predict(self, obs: torch.Tensor,
    #             deterministic: bool = False) -> torch.Tensor:
    #     """Predict action from observation.
    #
    #     deterministic is not used in this method, it is just for compatibility.
    #
    #     Args:
    #         obs (torch.Tensor): Observation.
    #         deterministic (bool, optional): Whether to return deterministic action. Defaults to False.
    #
    #     Returns:
    #         torch.Tensor: Action.
    #     """
    #
    #     device = self.inv_model.state_dict()['0.weight'].device
    #     # obs = obs.to(device)
    #     if obs.ndim == 1:
    #         obs = obs.unsqueeze(0).to(device)
    #     # history_obs = self.history_obs_update(obs.cpu()).to(device)
    #     # obs = torch.from_numpy(self.dataset_normalizer.normalize(obs.cpu().numpy(), 'observations')).to(device)
    #     if self.multi_step_count % self.multi_step_pred == 0:
    #         self.multi_step_count = 0
    #         # 配置生成条件参数，returns，constraints，skills
    #         returns = 0.95 * torch.ones(1, 1, device=device)
    #         constraints = torch.tensor([[1, 1]], dtype=torch.float32, device=device)
    #         skills = torch.tensor([[1, 0]], dtype=torch.float32, device=device)
    #         constraints = None
    #         skills = None
    #         samples = self.conditional_sample(obs, returns=returns, constraints=constraints, skills=skills)
    #         self.samples = samples.clone()
    #         self.samples.to(device)
    #     next_obs = self.samples[:, self.history_lenght + self.multi_step_count, :]
    #     obs_comb = torch.cat([obs, next_obs], dim=-1)
    #     obs_comb = obs_comb.reshape(-1, 2 * self.observation_dim)
    #     action = self.inv_model(obs_comb)
    #     self.multi_step_count += 1
    #     # action = torch.from_numpy(self.dataset_normalizer.unnormalize(action.detach().cpu().numpy(), 'actions')).to(device)
    #     return action.squeeze(0)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
