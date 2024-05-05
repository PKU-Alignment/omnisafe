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
"""Implementation of the Latent Model for Safe SLAC."""


from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def calculate_kl_divergence(
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
) -> torch.Tensor:
    """Calculate the KL divergence between two normal distributions.

    Args:
        p_mean (torch.Tensor): Mean of the first normal distribution.
        p_std (torch.Tensor): Standard deviation of the first normal distribution.
        q_mean (torch.Tensor): Mean of the second normal distribution.
        q_std (torch.Tensor): Standard deviation of the second normal distribution.

    Returns:
        torch.Tensor: The KL divergence between the two distributions.
    """
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_activation: nn.Module,
    hidden_sizes: list[int] | None = None,
) -> nn.Sequential:
    """Build a multi-layer perceptron (MLP) model using PyTorch.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output.
        hidden_sizes (list[int], optional): List of integers defining the number of units in each hidden layer.
        hidden_activation (nn.Module): Activation function to use after each hidden layer.

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    layers: list[Any] = []
    units = input_dim
    for next_units in hidden_sizes:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    model = nn.Sequential(*layers)
    model.add_module('last_linear', nn.Linear(units, output_dim))
    return model


def initialize_weight(m: nn.Module) -> None:
    """Initializes the weights of the module using Xavier uniform initialization.

    Args:
        m (nn.Module): The module whose weights need to be initialized.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FixedGaussian(nn.Module):
    """Represents a fixed diagonal Gaussian distribution.

    Attributes:
        output_dim (int): Dimension of the output distribution.
        std (float): Standard deviation of the Gaussian distribution.
    """

    def __init__(self, output_dim: int, std: float) -> None:
        """Initialize an instance of the Fixed Gaussian."""
        super().__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x: torch.Tensor) -> tuple:
        """Generates a mean and standard deviation tensor based on the fixed parameters.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Mean and standard deviation tensors.
        """
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device) * self.std
        return mean, std


class Gaussian(nn.Module):
    """Represents a diagonal Gaussian distribution with state dependent variances.

    Attributes:
        net (nn.Module): Neural network module to compute means and log standard deviations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
    ) -> None:
        """Initialize an instance of the Gaussian distribution."""
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_activation=nn.ELU(),
            hidden_sizes=hidden_sizes,
        ).apply(initialize_weight)

    def forward(self, x: torch.Tensor) -> tuple:
        """Computes the mean and standard deviation for the Gaussian distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Mean and log standard deviation tensors.
        """
        if x.ndim == 3:
            batch_size, seq_length, _ = x.size()
            x = self.net(x.view(batch_size * seq_length, _)).view(batch_size, seq_length, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5  # pylint: disable=not-callable
        return mean, std


class Bernoulli(nn.Module):
    """A module representing a Bernoulli distribution.

    This class builds a multi-layer perceptron (MLP) that outputs parameters for a Bernoulli
    distribution. The output of the MLP is transformed using a sigmoid function to ensure it lies
    between 0 and 1, representing probabilities.

    Attributes:
        net (nn.Module): The neural network that builds the Bernoulli distribution.

    Args:
        input_dim (int): The number of input features to the MLP.
        output_dim (int): The number of output features from the MLP.
        hidden_sizes (list[int, int]): The sizes of the hidden layers in the MLP. Defaults to
            (256, 256).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
    ) -> None:
        """Initializes the Bernoulli module with the specified architecture."""
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_activation=nn.ELU(),
            hidden_sizes=hidden_sizes,
        ).apply(initialize_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the Bernoulli module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing probabilities after applying the sigmoid function.
        """
        if x.ndim == 3:
            batch_size, seq_length, _ = x.size()
            x = self.net(x.view(batch_size * seq_length, -1)).view(batch_size, seq_length, -1)
        else:
            x = self.net(x)
        return torch.sigmoid(x)


class Decoder(torch.nn.Module):
    """The image processing decoder module.

    This decoder module that takes in a latent vector and outputs reconstructed images,
    also outputs a tensor of the same shape with a constant standard deviation value.

    Attributes:
        net (torch.nn.Sequential): The neural network layers.
        std (float): The standard deviation value to be used for the output tensor.

    Args:
        input_dim (int): Dimension of the input latent vector. Defaults to 288.
        output_dim (int): The number of output channels. Defaults to 3.
        std (float): Standard deviation for the generated tensor. Defaults to 1.0.
    """

    def __init__(self, input_dim: int = 288, output_dim: int = 3, std: float = 1.0) -> None:
        """Initializes the decoder module."""
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)
        self.std = std

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass for generating output image tensor and a tensor filled with std.

        Args:
            x (torch.Tensor): Input latent vector tensor.

        Returns:
            tuple: A tuple containing the reconstructed image tensor and a tensor filled with std.
        """
        batch_size, seq_length, latent_dim = x.size()
        x = x.view(batch_size * seq_length, latent_dim, 1, 1)
        x = self.net(x)
        _, channels, width, height = x.size()
        x = x.view(batch_size, seq_length, channels, width, height)
        return x, torch.ones_like(x).mul_(self.std)


class Encoder(torch.nn.Module):
    """An encoder module that takes in images and outputs a latent vector representation.

    Attributes:
        net (torch.nn.Sequential): The neural network layers.

    Args:
        input_dim (int): Number of input channels in the image. Defaults to 3.
        output_dim (int): Dimension of the output latent vector. Defaults to 256.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 256) -> None:
        """Initialize the Encoder module."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, output_dim, 4),
            nn.ELU(inplace=True),
        ).apply(initialize_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to transform input images into a flat latent vector representation.

        Args:
            x (torch.Tensor): Input tensor of images.

        Returns:
            torch.Tensor: Output tensor of latent vectors.
        """
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        x = self.net(x)
        return x.view(batch_size, seq_length, -1)


# pylint: disable-next=too-many-instance-attributes
class CostLatentModel(nn.Module):
    """The latent model for cost prediction.

    Stochastic latent variable model that estimates latent dynamics, rewards, and costs
    for a given observation and action space using variational inference techniques.

    Args:
        obs_shape (tuple of int): Shape of the observations.
        act_shape (tuple of int): Shape of the actions.
        feature_dim (int): Dimension of the feature vector from observations. Defaults to 256.
        latent_dim_1 (int): Dimension of the first set of latent variables. Defaults to 32.
        latent_dim_2 (int): Dimension of the second set of latent variables. Defaults to 256.
        hidden_sizes (list of int): Sizes of hidden layers in the networks.
        image_noise (float): Standard deviation of noise in image reconstruction. Defaults to 0.1.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        hidden_sizes: list[int],
        feature_dim: int = 256,
        latent_dim_1: int = 32,
        latent_dim_2: int = 256,
        image_noise: float = 0.1,
    ) -> None:
        """Initialize the instance of CostLatentModel."""
        super().__init__()
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.z1_prior_init = FixedGaussian(latent_dim_1, 1.0)
        self.z2_prior_init = Gaussian(latent_dim_1, latent_dim_2, hidden_sizes)
        self.z1_prior = Gaussian(
            int(latent_dim_2 + act_shape[0]),
            int(latent_dim_1),
            hidden_sizes,
        )
        self.z2_prior = Gaussian(
            int(latent_dim_1 + latent_dim_2 + act_shape[0]),
            int(latent_dim_2),
            hidden_sizes,
        )

        self.z1_posterior_init = Gaussian(feature_dim, latent_dim_1, hidden_sizes)
        self.z2_posterior_init = self.z2_prior_init
        self.z1_posterior = Gaussian(
            int(feature_dim + latent_dim_2 + act_shape[0]),
            int(latent_dim_1),
            hidden_sizes,
        )
        self.z2_posterior = self.z2_prior

        self.reward = Gaussian(
            int(2 * latent_dim_1 + 2 * latent_dim_2 + act_shape[0]),
            1,
            hidden_sizes,
        )

        self.cost = Bernoulli(
            int(2 * latent_dim_1 + 2 * latent_dim_2 + act_shape[0]),
            1,
            hidden_sizes,
        )

        self.encoder = Encoder(obs_shape[0], feature_dim)
        self.decoder = Decoder(
            int(latent_dim_1 + latent_dim_2),
            int(obs_shape[0]),
            std=np.sqrt(image_noise),
        )
        self.apply(initialize_weight)

    def sample_prior(self, actions_: torch.Tensor, z2_post_: torch.Tensor) -> tuple:
        """Sample the prior distribution for latent variables using initial and recurrent models.

        Args:
            actions_ (torch.Tensor): The actions taken at each step of the sequence.
            z2_post_ (torch.Tensor): The posterior samples of the second set of latent variables.

        Returns:
            tuple: A tuple containing means and standard deviations for the first set of latent variables.
        """
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        z1_mean_, z1_std_ = self.z1_prior(
            torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1),
        )
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    def sample_posterior(self, features_: torch.Tensor, actions_: torch.Tensor) -> tuple:
        """Sample the posterior distribution for latent variables.

        Args:
            features_ (torch.Tensor): The features extracted from observations at each timestep.
            actions_ (torch.Tensor): The actions taken at each step of the sequence.

        Returns:
            tuple: A tuple of tensors containing means, standard deviations, and samples of the latent variables.
        """
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            z1_mean, z1_std = self.z1_posterior(
                torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1),
            )
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
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

    # pylint: disable-next=too-many-locals
    def calculate_loss(
        self,
        state_: torch.Tensor,
        action_: torch.Tensor,
        reward_: torch.Tensor,
        done_: torch.Tensor,
        cost_: torch.Tensor,
    ) -> tuple:
        """Calculate the loss for the model.

        Args:
            state_ (torch.Tensor): Observed states over a sequence of timesteps.
            action_ (torch.Tensor): Actions taken at each timestep.
            reward_ (torch.Tensor): Observed rewards at each timestep.
            done_ (torch.Tensor): Done flags indicating the end of an episode.
            cost_ (torch.Tensor): Observed costs at each timestep.

        Returns:
            tuple: A tuple containing the KL divergence loss, image reconstruction loss,
                   reward prediction loss, and cost classification loss.
        """
        feature_ = self.forward(state_)

        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        loss_kld = (
            calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_)
            .mean(dim=0)
            .sum()
        )

        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(
            2 * math.pi,
        )
        loss_image = -log_likelihood_.mean(dim=0).sum()

        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        batch_size, seq_length, concated_shape = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(batch_size * seq_length, concated_shape))
        reward_mean_ = reward_mean_.view(batch_size, seq_length, 1)
        reward_std_ = reward_std_.view(batch_size, seq_length, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(
            2 * math.pi,
        )
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()

        p = self.cost(x.view(batch_size * seq_length, concated_shape)).view(
            batch_size,
            seq_length,
            1,
        )
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the encoded state of observation."""
        return self.encoder(obs)
