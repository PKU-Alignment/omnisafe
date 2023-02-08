# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of CholeskyActor."""

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal

from omnisafe.utils.model_utils import Activation, InitFunction, build_mlp_network, initialize_layer


# pylint: disable-next=too-many-instance-attributes
class MLPCholeskyActor(nn.Module):
    r"""Implementation of CholeskyActor.

    A Gaussian policy that uses a MLP to map observations to actions distributions.
    :class:`MLPCholeskyActor` uses a double headed MLP ,
    to predict the mean and Cholesky decomposition of the Gaussian distribution.

    .. note::
        The Cholesky decomposition is a lower triangular matrix L with positive diagonal entries,
        such that :math:`L^T L = \Sigma`, where :math:`\Sigma` is the covariance matrix of the Gaussian distribution.
        The Cholesky decomposition is a convenient way to represent a covariance matrix,
        and it is more numerically stable than the standard representation of the covariance matrix.

    This class is an inherit class of :class:`Actor`.
    You can design your own Gaussian policy by inheriting this class or :class:`Actor`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_max: torch.Tensor,
        act_min: torch.Tensor,
        hidden_sizes: list,
        cov_min: float,
        mu_clamp_min: float,
        mu_clamp_max: float,
        cov_clamp_min: float,
        cov_clamp_max: float,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'xavier_uniform',
    ) -> None:
        """Initialize MLPCholeskyActor.

        Args:
            obs_dim (int): observation dimension.
            act_dim (int): action dimension.
            act_max (torch.Tensor): maximum value of the action.
            act_min (torch.Tensor): minimum value of the action.
            hidden_sizes (list): list of hidden layer sizes.
            activation (str): activation function.
            cov_min (float): minimum value of the covariance matrix.
            mu_clamp_min (float): minimum value of the mean.
            mu_clamp_max (float): maximum value of the mean.
            cov_clamp_min (float): minimum value of the covariance matrix.
            cov_clamp_max (float): maximum value of the covariance matrix.
            weight_initialization_mode (str): weight initialization mode.
        """
        super().__init__()
        pi_sizes = [obs_dim] + hidden_sizes
        self.act_limit = act_max
        self.act_low = torch.nn.Parameter(
            torch.as_tensor(act_min), requires_grad=False
        )  # (1, act_dim)
        self.act_high = torch.nn.Parameter(
            torch.as_tensor(act_max), requires_grad=False
        )  # (1, act_dim)
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.cov_min = cov_min
        self.mu_clamp_min = mu_clamp_min
        self.mu_clamp_max = mu_clamp_max
        self.cov_clamp_min = cov_clamp_min
        self.cov_clamp_max = cov_clamp_max

        self.net = build_mlp_network(pi_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.cholesky_layer = nn.Linear(hidden_sizes[-1], (self.act_dim * (self.act_dim + 1)) // 2)
        initialize_layer(weight_initialization_mode, self.mu_layer)
        # initialize_layer(weight_initialization_mode,self.cholesky_layer)
        nn.init.constant_(self.mu_layer.bias, 0.0)
        nn.init.constant_(self.cholesky_layer.bias, 0.0)

    def predict(
        self, obs: torch.Tensor, deterministic: bool = False, need_log_prob: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Predict action given observation.

        .. note::
            - Compute the mean and Cholesky decomposition of the Gaussian distribution.
            - Compute logprob from Gaussian, and then apply correction for Tanh squashing.
              For details of the correction formula,
              please refer to the original `SAC paper <https://arxiv.org/abs/1801.01290>`_.
            - Get action from Multi-variate Gaussian distribution.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool): Whether to use deterministic policy.
        """
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(obs, dim=0)
        obs_length = obs.size(0)

        net_out = self.net(obs)

        clamped_mu = torch.clamp(self.mu_layer(net_out), self.mu_clamp_min, self.mu_clamp_max)
        mean = torch.sigmoid(clamped_mu)  # (B, act_dim)

        mean = self.act_low + (self.act_high - self.act_low) * mean
        cholesky_vector = torch.clamp(
            self.cholesky_layer(net_out), self.cov_clamp_min, self.cov_clamp_max
        )
        cholesky_diag_index = torch.arange(self.act_dim, dtype=torch.long) + 1
        cholesky_diag_index = (
            torch.div(cholesky_diag_index * (cholesky_diag_index + 1), 2, rounding_mode='floor') - 1
        )
        cholesky_vector[:, cholesky_diag_index] = (
            F.softplus(cholesky_vector[:, cholesky_diag_index]) + self.cov_min
        )
        tril_indices = torch.tril_indices(row=self.act_dim, col=self.act_dim, offset=0)
        cholesky = torch.zeros(size=(obs_length, self.act_dim, self.act_dim), dtype=torch.float32)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        pi_distribution = MultivariateNormal(mean.to(torch.float32), scale_tril=cholesky)

        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_distribution.rsample()

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if need_log_prob:
            return (
                pi_action.to(torch.float32),
                pi_action.to(torch.float32),
                cholesky.to(torch.float32),
            )
        return pi_action.to(torch.float32), pi_action.to(torch.float32)

    def forward(self, obs, deterministic=False):
        """Forward."""
