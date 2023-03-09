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
"""Implementation of CholeskyActor."""

from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal, Distribution
from omnisafe.models.base import Actor

from omnisafe.utils.model import Activation, InitFunction, build_mlp_network
from omnisafe.typing import OmnisafeSpace

# pylint: disable-next=too-many-instance-attributes
class CholeskyActor(Actor):
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
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        cov_min: float=1e-4,
        mean_clamp_min: float=-5,
        mean_clamp_max: float=5,
        cov_clamp_min: float=-5,
        cov_clamp_max: float=20,
    ) -> None:
        """Initialize MLPCholeskyActor.

        Args:
            obs_space (int): observation dimension.
            act_space (int): action dimension.
            act_max (torch.Tensor): maximum value of the action.
            act_min (torch.Tensor): minimum value of the action.
            hidden_sizes (list): list of hidden layer sizes.
            activation (str): activation function.
            cov_min (float): minimum value of the covariance matrix.
            mean_clamp_min (float): minimum value of the mean.
            mean_clamp_max (float): maximum value of the mean.
            cov_clamp_min (float): minimum value of the covariance matrix.
            cov_clamp_max (float): maximum value of the covariance matrix.
            weight_initialization_mode (str): weight initialization mode.
        """
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.cov_min = cov_min
        self.mean_clamp_min = mean_clamp_min
        self.mean_clamp_max = mean_clamp_max
        self.cov_clamp_min = cov_clamp_min
        self.cov_clamp_max = cov_clamp_max
        self.net = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim * 2],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        act_dim = self._act_space.shape[0]
        mean, cholesky_vector = self.net(obs).chunk(2, dim=-1)
        mean = torch.clamp(mean, min=self.mean_clamp_min, max=self.mean_clamp_max)
        mean = torch.sigmoid(mean)
        cholesky_vector = torch.clamp(cholesky_vector, min=self.cov_clamp_min, max=self.cov_clamp_max)
        cholesky_diag_index = torch.arange(act_dim, dtype=torch.long) + 1
        cholesky_diag_index = torch.div(
            cholesky_diag_index *
            (cholesky_diag_index + 1), 2, rounding_mode='floor') - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(
            cholesky_vector[:, cholesky_diag_index]) + self.cov_min
        tril_indices = torch.tril_indices(row=act_dim, col=act_dim, offset=0)
        cholesky = torch.zeros(size=(obs.size(0), act_dim, act_dim), dtype=torch.float32)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return MultivariateNormal(loc=mean, scale_tril=cholesky)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

    def forward(self, obs: torch.Tensor) -> Distribution:
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('CholeksyActor does not support log_prob')