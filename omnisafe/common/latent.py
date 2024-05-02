import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def build_mlp(
    input_dim,
    output_dim,
    hidden_sizes=None,
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    layers = []
    units = input_dim
    for next_units in hidden_sizes:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    model = nn.Sequential(*layers)
    model.add_module('last_linear', nn.Linear(units, output_dim))
    if output_activation is not None:
        model.add_module('output_activation', output_activation)
    return model


def initialize_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FixedGaussian(torch.nn.Module):
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, output_dim, std) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(torch.nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std


class Bernoulli(torch.nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        return torch.sigmoid(x)


class Decoder(torch.nn.Module):
    """
    Decoder.
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)
        self.std = std

    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)


class Encoder(torch.nn.Module):
    """
    Encoder.
    """

    def __init__(self, input_dim=3, output_dim=256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.ELU(inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ELU(inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.ELU(inplace=True),
        ).apply(initialize_weight)

    def forward(self, x):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x = self.net(x)
        return x.view(B, S, -1)


class CostLatentModel(torch.nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics, reward and cost.
    """

    def __init__(
        self,
        obs_shape,
        act_shape,
        feature_dim=256,
        latent_dim_1=32,
        latent_dim_2=256,
        hidden_sizes=(256, 256),
        image_noise=0.1,
    ) -> None:
        super().__init__()
        self.bceloss = torch.nn.BCELoss(reduction='none')
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(latent_dim_1, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(latent_dim_1, latent_dim_2, hidden_sizes)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            latent_dim_2 + act_shape[0],
            latent_dim_1,
            hidden_sizes,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            latent_dim_1 + latent_dim_2 + act_shape[0],
            latent_dim_2,
            hidden_sizes,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(feature_dim, latent_dim_1, hidden_sizes)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            feature_dim + latent_dim_2 + act_shape[0],
            latent_dim_1,
            hidden_sizes,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * latent_dim_1 + 2 * latent_dim_2 + act_shape[0],
            1,
            hidden_sizes,
        )

        self.cost = Bernoulli(
            2 * latent_dim_1 + 2 * latent_dim_2 + act_shape[0],
            1,
            hidden_sizes,
        )

        # feat(t) = Encoder(x(t))
        self.encoder = Encoder(obs_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            latent_dim_1 + latent_dim_2,
            obs_shape[0],
            std=np.sqrt(image_noise),
        )
        self.apply(initialize_weight)

    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(
            torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1)
        )
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(
                torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1)
            )
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    #
    def calculate_loss(self, state_, action_, reward_, done_, cost_):
        # Calculate the sequence of features.
        feature_ = self.encoder(state_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = (
            calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_)
            .mean(dim=0)
            .sum()
        )

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(
            2 * math.pi
        )
        loss_image = -log_likelihood_.mean(dim=0).sum()

        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(
            2 * math.pi
        )
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()

        p = self.cost(x.view(B * S, X)).view(B, S, 1)
        q = 1 - p
        weight_p = 100
        binary_cost_ = torch.sign(cost_)
        loss_cost = (
            -30
            * (
                weight_p * binary_cost_ * torch.log(p + 1e-6)
                + (1 - binary_cost_) * torch.log(q + 1e-6)
            )
            .mean(dim=0)
            .sum()
        )

        return loss_kld, loss_image, loss_reward, loss_cost
