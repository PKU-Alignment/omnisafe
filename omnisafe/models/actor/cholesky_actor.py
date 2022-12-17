# Copyright 2022 OmniSafe Team. All Rights Reserved.
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

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal

from omnisafe.utils.model_utils import build_mlp_network, initialize_layer


class MLPCholeskyActor(nn.Module):

    COV_MIN = 1e-4  # last exp is 1e-2
    MEAN_CLAMP_MIN = -5
    MEAN_CLAMP_MAX = 5
    COV_CLAMP_MIN = -5
    COV_CLAMP_MAX = 20

    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        hidden_sizes,
        activation,
        cov_min,
        mu_clamp_min,
        mu_clamp_max,
        cov_clamp_min,
        cov_clamp_max,
        weight_initialization_mode,
        shared=None,
    ):
        super().__init__()
        pi_sizes = [obs_dim] + hidden_sizes
        self.act_limit = act_limit
        self.act_low = torch.nn.Parameter(
            torch.as_tensor(-act_limit), requires_grad=False
        )  # (1, act_dim)
        self.act_high = torch.nn.Parameter(
            torch.as_tensor(act_limit), requires_grad=False
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

    def predict(self, obs, determinstic=False):
        """
        forwards input through the network
        :param obs: (B, obs_dim)
        :return: mu vector (B, act_dim) and cholesky factorization of covariance matrix (B, act_dim, act_dim)
        """
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(obs, dim=0)
        B = obs.size(0)

        net_out = self.net(obs)

        clamped_mu = torch.clamp(self.mu_layer(net_out), self.mu_clamp_min, self.mu_clamp_max)
        mu = torch.sigmoid(clamped_mu)  # (B, act_dim)

        mu = self.act_low + (self.act_high - self.act_low) * mu
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        cholesky_vector = torch.clamp(
            self.cholesky_layer(net_out), self.cov_clamp_min, self.cov_clamp_max
        )  # (B, (act_dim*(act_dim+1))//2)
        cholesky_diag_index = torch.arange(self.act_dim, dtype=torch.long) + 1
        # cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_diag_index = (
            torch.div(cholesky_diag_index * (cholesky_diag_index + 1), 2, rounding_mode='floor') - 1
        )
        # add a small value to prevent the diagonal from being 0.
        cholesky_vector[:, cholesky_diag_index] = (
            F.softplus(cholesky_vector[:, cholesky_diag_index]) + self.COV_MIN
        )
        tril_indices = torch.tril_indices(row=self.act_dim, col=self.act_dim, offset=0)
        cholesky = torch.zeros(size=(B, self.act_dim, self.act_dim), dtype=torch.float32)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        pi_distribution = MultivariateNormal(mu, scale_tril=cholesky)

        if determinstic == False:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action.squeeze(), cholesky
