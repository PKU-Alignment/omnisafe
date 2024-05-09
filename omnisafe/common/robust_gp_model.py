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
"""Implementation of Dynamics Model Based on GPyTorch."""
# mypy: ignore-errors


from __future__ import annotations

import os
import warnings
from typing import Callable

import gpytorch
import gymnasium as gym
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ZeroMean
from gpytorch.priors import NormalPrior

from omnisafe.typing import DEVICE_CPU
from omnisafe.utils.tools import to_tensor


DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2}}
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1]}


class BaseGPy(gpytorch.models.ExactGP):
    """A Gaussian Process (GP) model using a zero mean function and a scaled RBF kernel with priors.

    This class extends gpytorch.models.ExactGP, specifically designed for use in
    disturbance estimation tasks.

    Attributes:
        mean_module (ZeroMean): The mean module which is set to zero mean.
        covar_module (ScaleKernel): The covariance kernel, a scaled RBF kernel with specified priors.

    Args:
        train_x (Tensor): Training input features, which should be a tensor.
        train_y (Tensor): Training target values, which should be a tensor.
        prior_std (float): The prior standard deviation used to adjust the output scale of the kernel.
        likelihood (Likelihood): The likelihood function associated with the GP model.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        prior_std: float,
        likelihood: Likelihood,
    ) -> None:
        """Initialize the BaseGPy model."""
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=NormalPrior(1e5, 1e-5)),
            outputscale_prior=NormalPrior(prior_std + 1e-6, 1e-5),
        )
        self.covar_module.base_kernel.lengthscale = 1e5
        self.covar_module.outputscale = prior_std + 1e-6

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through the GP model to produce a multivariate normal distribution.

        Args:
            x (Tensor): Input features for which predictions are to be made.

        Returns:
            MultivariateNormal: A multivariate normal distribution reflecting the GP predictions.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class GPyDisturbanceEstimator:
    """A class for estimating disturbances using Gaussian Processes with GPyTorch.

    Attributes:
        device (torch.device): The device (CPU or CUDA) on which the tensors will be processed.
        _train_x (torch.Tensor): Training data features.
        _train_y (torch.Tensor): Training data targets.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood model for GP inference.
        model (BaseGPy): The GPyTorch model.

    Args:
        train_x (torch.Tensor): Training data features. If not a tensor, it will be converted.
        train_y (torch.Tensor): Training data targets. If not a tensor, it will be converted.
        prior_std (float): Standard deviation of the prior distribution.
        likelihood (Optional[gpytorch.likelihoods.Likelihood]): A GPyTorch likelihood.
        device (Optional[torch.device]): The torch device. Defaults to CPU if None.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        prior_std: float,
        likelihood: gpytorch.likelihoods.Likelihood | None = None,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize the GPyDisturbanceEstimator."""
        self.device = device if device else torch.device('cpu')

        if not torch.is_tensor(train_x):
            train_x = torch.tensor(train_x, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(train_y):
            train_y = torch.tensor(train_y, dtype=torch.float32, device=self.device)
        self._train_x = train_x
        self._train_y = train_y

        if not likelihood:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood.to(self.device)

        self.model = BaseGPy(train_x, train_y, prior_std, likelihood)
        self.model = self.model.to(self.device)
        warnings.filterwarnings('ignore')

    def train(self, training_iter: int) -> None:
        """Trains the Gaussian Process model.

        Args:
            training_iter (int): Number of training iterations.
            verbose (bool): If True, prints detailed logging information.
        """
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self._train_x)
            loss = -mll(output, self._train_y)
            loss.backward()
            optimizer.step()

    def predict(self, test_x: torch.Tensor) -> dict[str, torch.Tensor | np.ndarray]:
        """Makes predictions on new data.

        Args:
            test_x (torch.Tensor): Test data features. If not a tensor, it will be converted.

        Returns:
            A dictionary containing prediction mean, variance, covariance matrix, and confidence
            intervals. If the input was not a tensor, values will be converted to numpy arrays.
        """
        is_tensor = torch.is_tensor(test_x)
        if not is_tensor:
            test_x = torch.tensor(test_x, dtype=torch.float32, device=self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            pred_dict = {
                'mean': observed_pred.mean.cpu(),
                'f_var': observed_pred.variance.cpu(),
                'f_covar': observed_pred.covariance_matrix.cpu(),
                'lower_ci': observed_pred.confidence_region()[0].cpu(),
                'upper_ci': observed_pred.confidence_region()[1].cpu(),
            }

        if not is_tensor:
            for key, val in pred_dict.items():
                pred_dict[key] = val.numpy()

        return pred_dict


# pylint: disable-next=too-many-instance-attributes
class DynamicsModel:
    """Initializes the DynamicsModel with a gym environment.

    Args:
        env (gym.Env): The gym environment to model dynamics for.
        gp_model_size (int, optional): Maximum history count for disturbances. Defaults to 2000.
        l_p (float, optional): Learning parameter. Defaults to 0.03.
        device (str, optional): The device to perform computations on. Defaults to 'cpu'.
    """

    def __init__(
        self,
        env: gym.Env,
        gp_model_size: int = 2000,
        l_p: float = 0.03,
        device: str = 'cpu',
    ) -> None:
        """Initializes the DynamicsModel with a gym environment."""
        self.env = env
        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE[self.env.dynamics_mode]['n_s']
        self.n_u = DYNAMICS_MODE[self.env.dynamics_mode]['n_u']

        self.disturbance_history = {}
        self.history_counter = 0
        self.max_history_count = gp_model_size
        self.disturbance_history['state'] = np.zeros((self.max_history_count, self.n_s))
        self.disturbance_history['disturbance'] = np.zeros((self.max_history_count, self.n_s))
        self._train_x = np.zeros((self.max_history_count, self.n_s))
        self._train_y = np.zeros((self.max_history_count, self.n_s))
        self._disturb_estimators = []
        self.device = torch.device(device)

        for i in range(self.n_s):
            self._disturb_estimators.append(
                GPyDisturbanceEstimator(
                    np.zeros((self.max_history_count, self.n_s)),
                    np.zeros((self.max_history_count, self.n_s)),
                    MAX_STD[self.env.dynamics_mode][i],
                    device=self.device,
                ),
            )
        self._disturb_initialized = True
        self.l_p = l_p

    def get_dynamics(self) -> tuple[Callable, Callable]:
        """Retrieves the dynamics functions for drift and control based on the environment's dynamics mode.

        Returns:
            tuple: A tuple containing two callable methods, `get_f` and `get_g`.
        """
        if self.env.dynamics_mode == 'Unicycle':

            def get_f(state_batch: np.ndarray) -> np.ndarray:
                return np.zeros(state_batch.shape)

            def get_g(state_batch: np.ndarray) -> np.ndarray:
                theta = state_batch[:, 2]
                g_x = np.zeros((state_batch.shape[0], 3, 2))
                g_x[:, 0, 0] = np.cos(theta)
                g_x[:, 1, 0] = np.sin(theta)
                g_x[:, 2, 1] = 1.0
                return g_x

        else:
            raise NotImplementedError('Unknown Dynamics mode.')

        return get_f, get_g

    def get_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Processes the raw observations from the environment.

        Args:
            obs (torch.Tensor): The environment observations.

        Returns:
            torch.Tensor: The processed state of the system.
        """
        expand_dims = len(obs.shape) == 1
        dtype = obs.dtype
        device = obs.device
        obs = obs.cpu().numpy() if obs.is_cuda else obs.numpy()

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        if self.env.dynamics_mode == 'Unicycle':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
        else:
            raise NotImplementedError('Unknown dynamics')

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return torch.tensor(state_batch, dtype=dtype, device=device)

    def append_transition(
        self,
        state_batch: np.ndarray,
        u_batch: np.ndarray,
        next_state_batch: np.ndarray,
    ) -> None:
        """Estimates the disturbance from the current dynamics transition and adds it to the buffer.

        Args:
            state_batch (np.ndarray): The batch of current states, shape (n_s,) or (batch_size, n_s).
            u_batch (np.ndarray): The batch of actions applied, shape (n_u,) or (batch_size, n_u).
            next_state_batch (np.ndarray): The batch of next states, shape (n_s,) or (batch_size, n_s).
        """
        u_batch = np.expand_dims(u_batch, -1)
        disturbance_batch = (
            next_state_batch
            - state_batch
            - self.env.dt
            * (self.get_f(state_batch) + (self.get_g(state_batch) @ u_batch).squeeze(-1))
        ) / self.env.dt

        for i in range(state_batch.shape[0]):
            self.disturbance_history['state'][self.history_counter % self.max_history_count] = (
                state_batch[i]
            )
            self.disturbance_history['disturbance'][
                self.history_counter % self.max_history_count
            ] = disturbance_batch[i]
            self.history_counter += 1

            if self.history_counter % (self.max_history_count // 10) == 0:
                self.fit_gp_model()

    def fit_gp_model(self, training_iter: int = 70) -> None:
        """Fits a Gaussian Process model to the disturbance data.

        Args:
            training_iter (int, optional): Number of training iterations for the GP model. Defaults to 70.
        """
        if self.history_counter < self.max_history_count:
            train_x = self.disturbance_history['state'][: self.history_counter]
            train_y = self.disturbance_history['disturbance'][: self.history_counter]
        else:
            train_x = self.disturbance_history['state']
            train_y = self.disturbance_history['disturbance']

        train_x_std = np.std(train_x, axis=0)
        train_x_normalized = train_x / (train_x_std + 1e-8)
        train_y_std = np.std(train_y, axis=0)
        train_y_normalized = train_y / (train_y_std + 1e-8)

        self._disturb_estimators = []
        for i in range(self.n_s):
            self._disturb_estimators.append(
                GPyDisturbanceEstimator(
                    train_x_normalized,
                    train_y_normalized[:, i],
                    MAX_STD[self.env.dynamics_mode][i],
                    device=self.device,
                ),
            )
            self._disturb_estimators[i].train(training_iter)
        self._disturb_initialized = False
        self._train_x = train_x
        self._train_y = train_y

    def predict_disturbance(self, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the disturbance at the queried states using the trained Gaussian Process models.

        Args:
            test_x (torch.Tensor): The state for which to predict disturbances, shape (n_test, n_s).

        Returns:
            tuple: A tuple of arrays (means, variances).
        """
        dtype = test_x.dtype
        device = test_x.device
        test_x = test_x.cpu().detach().double().numpy()

        expand_dims = len(test_x.shape) == 1
        if expand_dims:
            test_x = np.expand_dims(test_x, axis=0)

        means = np.zeros(test_x.shape)
        f_std = np.zeros(test_x.shape)

        if not self._disturb_initialized:
            train_x_std = np.std(self._train_x, axis=0)
            train_y_std = np.std(self._train_y, axis=0)
            test_x = test_x / train_x_std
            for i in range(self.n_s):
                prediction_ = self._disturb_estimators[i].predict(test_x)
                means[:, i] = prediction_['mean'] * (train_y_std[i] + 1e-8)
                f_std[:, i] = np.sqrt(prediction_['f_var']) * (train_y_std[i] + 1e-8)

        else:
            f_std = np.ones(test_x.shape)
            for i in range(self.n_s):
                f_std[:, i] *= MAX_STD[self.env.dynamics_mode][i]

        if expand_dims:
            means = means.squeeze(0)
            f_std = f_std.squeeze(0)

        return (to_tensor(means, dtype, device), to_tensor(f_std, dtype, device))

    def load_disturbance_models(self, save_dir: str, epoch: str) -> None:
        """Loads the disturbance models and their training data.

        Args:
            save_dir (str): The directory where the model files are saved.
            epoch (str): The epoch identifier used in the filenames to load the specific model checkpoint.
        """
        self._disturb_estimators = []
        weights = torch.load(
            os.path.join(save_dir, f'gp_models_{epoch}.pkl'),
            map_location=self.device,
        )
        self._train_x = torch.load(os.path.join(save_dir, f'gp_models_train_x_{epoch}.pkl'))
        self._train_y = torch.load(os.path.join(save_dir, f'gp_models_train_y_{epoch}.pkl'))
        for i in range(self.n_s):
            self._disturb_estimators.append(
                GPyDisturbanceEstimator(
                    self._train_x,
                    self._train_y[:, i],
                    MAX_STD[self.env.dynamics_mode][i],
                    device=self.device,
                ),
            )
            self._disturb_estimators[i].model.load_state_dict(weights[i])

    @property
    def train_x(self) -> np.ndarray:
        """Returns the training data input features used for the disturbance estimators."""
        return self._train_x

    @property
    def train_y(self) -> np.ndarray:
        """Returns the training data labels used for the disturbance estimators."""
        return self._train_y

    @property
    def disturb_estimators(self) -> list[GPyDisturbanceEstimator]:
        """Provides access to the list of trained disturbance estimator models."""
        return self._disturb_estimators
