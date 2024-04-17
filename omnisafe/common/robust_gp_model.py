""" Adapted almost directly from:
https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html

Training is performed rapidly (and exactly) using GPUs and prediction is done very rapidly using LOVE.
"""

import torch
import numpy as np
import gpytorch
import warnings
warnings.filterwarnings('ignore')
from omnisafe.common.utils import to_tensor, to_numpy

DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},   # state = [x y θ]
                 'SimulatedCars': {'n_s': 10, 'n_u': 1},  # state = [x y θ v ω]
                 'Pvtol': {'n_s': 6, 'n_u': 2},  # state = [x y θ v_x v_y thrust]
                 'Pendulum': {'n_s': 2, 'n_u': 1}
                 }
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1], 'SimulatedCars': [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2],  'Pvtol': [0, 0, 0, 0, 0, 0],  'Pendulum': [0.1, 0.1, 0.1]}


class BaseGPy(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, prior_std, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.NormalPrior(1e5, 1e-5)),
                            outputscale_prior=gpytorch.priors.NormalPrior(prior_std + 1e-6, 1e-5))
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = 1e5
        self.covar_module.outputscale = prior_std + 1e-6

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GPyDisturbanceEstimator:
    """
    A wrapper around teh BaseGPy model above.
    """

    def __init__(self, train_x, train_y, prior_std, likelihood=None, device=None):

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        if not torch.is_tensor(train_x):
            train_x = to_tensor(train_x, torch.FloatTensor, self.device)
        if not torch.is_tensor(train_y):
            train_y = to_tensor(train_y, torch.FloatTensor, self.device)
        self.train_x = train_x
        self.train_y = train_y

        if not likelihood:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood.to(self.device)

        self.model = BaseGPy(train_x, train_y, prior_std, likelihood)
        self.model = self.model.to(self.device)

    def train(self, training_iter, verbose=False):

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose:
                print('\tIter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            optimizer.step()

    def predict(self, test_x):

        # Convert to torch tensor
        is_tensor = torch.is_tensor(test_x)
        if not is_tensor:
           test_x = to_tensor(test_x, torch.FloatTensor, self.device)

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            pred_dict = dict()
            pred_dict['mean'] = observed_pred.mean.cpu()
            pred_dict['f_var'] = observed_pred.variance.cpu()
            pred_dict['f_covar'] = observed_pred.covariance_matrix.cpu()
            lower_ci, upper_ci = observed_pred.confidence_region()
            pred_dict['lower_ci'] = lower_ci.cpu()
            pred_dict['upper_ci'] = upper_ci.cpu()

        # If they gave us ndarray, we give back ndarray
        if not is_tensor:
            for key, val in pred_dict.items():
                pred_dict[key] = to_numpy(val)

        return pred_dict

class DynamicsModel:

    def __init__(self, env, gp_model_size=2000, l_p=0.03, device='cpu'):
        """Constructor of DynamicsModel.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        """

        self.env = env
        # Get Dynamics
        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE[self.env.dynamics_mode]['n_s']
        self.n_u = DYNAMICS_MODE[self.env.dynamics_mode]['n_u']

        # Keep Disturbance History to estimate it using GPs
        self.disturb_estimators = None
        self.disturbance_history = dict()
        self.history_counter = 0  # keeping only max_history_count points in the buffer
        self.max_history_count = gp_model_size  # How many points we want to have in the GP
        self.disturbance_history['state'] = np.zeros((self.max_history_count, self.n_s))
        self.disturbance_history['disturbance'] = np.zeros((self.max_history_count, self.n_s))
        self.train_x = None  # x-data used to fit the last GP models
        self.train_y = None  # y-data used to fit the last GP models

        self.l_p = l_p

        self.device = torch.device(device)

    def predict_next_state(self, state_batch, u_batch, t_batch=None, use_gps=True):
        """Given the current state and action, this function predicts the next state.

        Parameters
        ----------
        state_batch : ndarray
            State
        u_batch : ndarray
            Action
        t_batch: ndarray, optional
            Time batch for state dependant dynamics
        use_gps : bool, optional
            Use GPs to return mean and var

        Returns
        -------
        next_state : ndarray
            Next state
        """

        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = np.expand_dims(state_batch, axis=0)

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        if t_batch is not None:
            next_state_batch = state_batch + self.env.dt * (self.get_f(state_batch, t_batch) + (self.get_g(state_batch, t_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))
        else:
            next_state_batch = state_batch + self.env.dt * (self.get_f(state_batch) + (self.get_g(state_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))

        if use_gps:  # if we want estimate the disturbance, let's do it!
            pred_mean, pred_std = self.predict_disturbance(state_batch)
            next_state_batch += self.env.dt * pred_mean
        else:
            pred_std = np.zeros(state_batch.shape)

        if expand_dims:
            next_state_batch = next_state_batch.squeeze(0)
            if pred_std is not None:
                pred_std = pred_std.squeeze(0)

        if t_batch is not None:
            next_t_batch = t_batch + self.env.dt
            return next_state_batch, self.env.dt * pred_std, next_t_batch

        return next_state_batch, self.env.dt * pred_std, t_batch

    def predict_next_obs(self, state, u):
        """Predicts the next observation given the state and u. Note that this only predicts the mean next observation.

        Parameters
        ----------
        state : ndarray
        u : ndarray

        Returns
        -------
        next_obs : ndarray
            Next observation
        """

        next_state, _, _ = self.predict_next_state(state, u)
        next_obs = self.get_obs(next_state)
        return next_obs

    def get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        if self.env.dynamics_mode == 'Unicycle':

            def get_f(state_batch, t_batch=None):
                f_x = np.zeros(state_batch.shape)
                return f_x

            def get_g(state_batch, t_batch=None):
                theta = state_batch[:, 2]
                g_x = np.zeros((state_batch.shape[0], 3, 2))
                g_x[:, 0, 0] = np.cos(theta)
                g_x[:, 1, 0] = np.sin(theta)
                g_x[:, 2, 1] = 1.0
                return g_x

        elif self.env.dynamics_mode == 'Pendulum':
            
            def get_f(state_batch, t_batch=None):
                f_x = np.zeros(state_batch.shape)
                theta = state_batch[:, 0]
                theta_dot = state_batch[:, 1]
                f_x = np.array(
                    [
                        -3*10/2*np.sin(theta+np.pi)*self.env.dt + theta,
                        theta_dot - 3*10/2*np.sin(theta+np.pi)
                    ]
                )
                return f_x

            def get_g(state_batch, t_batch=None):
                g_x = np.zeros((state_batch.shape[0], 2, 1))
                g_x[:, 0, 0] = 3*self.env.dt**2
                g_x[:, 1, 0] = 3*self.env.dt
                return g_x

        else:
            raise Exception('Unknown Dynamics mode.')

        return get_f, get_g

    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs_batch : ndarray or torch.tensor
            Environment observation.

        Returns
        -------
        state_batch : ndarray or torch.tensor
            State of the system.

        """

        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        if self.env.dynamics_mode == 'Unicycle':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
        elif self.env.dynamics_mode == 'Pendulum':
            theta = np.arctan2(obs[:, 1], obs[:, 0])
            theta_dot = obs[:, 2]
            state_batch = np.zeros((obs.shape[0], 2))
            state_batch[:, 0] = theta
            state_batch[:, 1] = theta_dot
        else:
            raise Exception('Unknown dynamics')

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch

    def get_obs(self, state_batch):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------
        state : ndarray
            Environment state batch of shape (batch_size, n_s)

        Returns
        -------
        obs : ndarray
          Observation batch of shape (batch_size, n_o)

        """

        if self.env.dynamics_mode == 'Unicycle':
            obs = np.zeros((state_batch.shape[0], 4))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])
            obs[:, 3] = np.sin(state_batch[:, 2])
        else:
            raise Exception('Unknown dynamics')
        return obs

    def append_transition(self, state_batch, u_batch, next_state_batch, t_batch=None):
        """Estimates the disturbance from the current dynamics transition and adds it to buffer.

        Parameters
        ----------
        state_batch : ndarray
            shape (n_s,) or (batch_size, n_s)
        u_batch : ndarray
            shape (n_u,) or (batch_size, n_u)
        next_state_batch : ndarray
            shape (n_s,) or (batch_size, n_s)
        t_batch : ndarray, optional
            shape (1,) or (batch_size, 1)

        Returns
        -------

        """

        expand_dims = len(state_batch.shape) == 1

        if expand_dims:
            state_batch = np.expand_dims(state_batch, 0)
            next_state_batch = np.expand_dims(next_state_batch, 0)
            u_batch = np.expand_dims(u_batch, 0)

        u_batch = np.expand_dims(u_batch, -1)  # for broadcasting batch matrix multiplication 
        disturbance_batch = (next_state_batch - state_batch - self.env.dt * (self.get_f(state_batch, t_batch) + (self.get_g(state_batch, t_batch) @ u_batch).squeeze(-1))) / self.env.dt

        # Append new data point (state, disturbance) to our dataset
        for i in range(state_batch.shape[0]):

            self.disturbance_history['state'][self.history_counter % self.max_history_count] = state_batch[i]
            self.disturbance_history['disturbance'][self.history_counter % self.max_history_count] = disturbance_batch[i]

            # Increment how many data points we have
            self.history_counter += 1

            # Update GP models every max_history_count data points
            if self.history_counter % (self.max_history_count/10) == 0:
                self.fit_gp_model()

    def fit_gp_model(self, training_iter=70):
        """

        Parameters
        ----------
        training_iter : int
            Number of training iterations for GP model.

        Returns
        -------

        """

        if self.history_counter < self.max_history_count:  # didn't fill the buffer yet
            train_x = self.disturbance_history['state'][:self.history_counter]
            train_y = self.disturbance_history['disturbance'][:self.history_counter]
        else:  # buffer filled, use all the data points
            train_x = self.disturbance_history['state']
            train_y = self.disturbance_history['disturbance']

        # Normalize Data
        train_x_std = np.std(train_x, axis=0)
        train_x_normalized = train_x / (train_x_std + 1e-8)
        train_y_std = np.std(train_y, axis=0)
        train_y_normalized = train_y / (train_y_std + 1e-8)

        self.disturb_estimators = []
        for i in range(self.n_s):
            # self.disturb_estimators.append(GPyDisturbanceEstimator(train_x, train_y[:, i]))
            self.disturb_estimators.append(GPyDisturbanceEstimator(train_x_normalized, train_y_normalized[:, i], MAX_STD[self.env.dynamics_mode][i], device=self.device))
            self.disturb_estimators[i].train(training_iter)

        # track the data I last used to fit the GPs for saving purposes (need it to initialize before loading weights)
        self.train_x = train_x
        self.train_y = train_y

    def predict_disturbance(self, test_x):
        """Predict the disturbance at the queried states using the GP models.

        Parameters
        ----------
        test_x : ndarray or torch.tensor
                shape(n_test, n_s)
        Returns
        -------
        means: ndarray or torch.tensor
            Prediction means -- shape(n_test, n_s)
        vars: ndarray or torch.tensor
            Prediction variances -- shape(n_test, n_s)
        """

        is_tensor = torch.is_tensor(test_x)

        if is_tensor:
            dtype = test_x.dtype
            device = test_x.device
            test_x = to_numpy(test_x)

        expand_dims = len(test_x.shape) == 1
        if expand_dims:
            test_x = np.expand_dims(test_x, axis=0)

        means = np.zeros(test_x.shape)
        f_std = np.zeros(test_x.shape)  # standard deviation

        if self.disturb_estimators:
            # Normalize
            train_x_std = np.std(self.train_x, axis=0)
            train_y_std = np.std(self.train_y, axis=0)
            test_x = test_x / train_x_std
            for i in range(self.n_s):
                prediction_ = self.disturb_estimators[i].predict(test_x)
                means[:, i] = prediction_['mean'] * (train_y_std[i] + 1e-8)
                f_std[:, i] = np.sqrt(prediction_['f_var']) * (train_y_std[i] + 1e-8)

        else:  # zero-mean, max_sigma prior
            f_std = np.ones(test_x.shape)
            for i in range(self.n_s):
                f_std[:, i] *= MAX_STD[self.env.dynamics_mode][i]

        if expand_dims:
            means = means.squeeze(0)
            f_std = f_std.squeeze(0)

        return (to_tensor(means, dtype, device), to_tensor(f_std, dtype, device)) if is_tensor else (means, f_std)

    def load_disturbance_models(self, output):

        if output is None:
            return

        self.disturb_estimators = []

        weights = torch.load('{}/gp_models.pkl'.format(output), map_location=self.device)
        self.train_x = torch.load('{}/gp_models_train_x.pkl'.format(output))
        self.train_y = torch.load('{}/gp_models_train_y.pkl'.format(output))
        for i in range(self.n_s):
            self.disturb_estimators.append(GPyDisturbanceEstimator(self.train_x, self.train_y[:, i], MAX_STD[self.env.dynamics_mode][i], device=self.device))
            self.disturb_estimators[i].model.load_state_dict(weights[i])

    def save_disturbance_models(self, output):

        if not self.disturb_estimators or self.train_x is None or self.train_y is None:
            return
        weights = []
        for i in range(len(self.disturb_estimators)):
            weights.append(self.disturb_estimators[i].model.state_dict())
        torch.save(weights, '{}/gp_models.pkl'.format(output))
        # Also save data used to fit model (needed for initializing the model before loading weights)
        torch.save(self.train_x, '{}/gp_models_train_x.pkl'.format(output))
        torch.save(self.train_y, '{}/gp_models_train_y.pkl'.format(output))

    def seed(self, seed):
        torch.manual_seed(seed)