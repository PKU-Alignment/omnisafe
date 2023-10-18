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
# Modified version of model.py from  https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py
# original version doesn't validate model error batch-wise and is highly memory intensive.
# ==============================================================================
"""The Dynamics Model of MBPO and PETS."""


from __future__ import annotations

import itertools
from collections import defaultdict
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnisafe.models.actor_critic import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.utils.config import Config


def swish(data: torch.Tensor) -> torch.Tensor:
    """Transform data using sigmoid function."""
    return data * torch.sigmoid(data)


class StandardScaler:
    """Normalizes data using standardization.

    This class provides methods to fit the scaler to the input data and transform
    the input data using the parameters learned during the fitting process.

    Args:
        device (torch.device): The device to use.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize an instance of :class:`StandardScaler`."""
        self._mean: float = 0.0
        self._std: float = 1.0
        self._mean_t: torch.Tensor = torch.tensor(self._mean).to(device)
        self._std_t: torch.Tensor = torch.tensor(self._std).to(device)
        self._device: torch.device = device

    def fit(self, data: np.ndarray) -> None:
        """Fits the scaler to the input data.

        Args:
            data (np.ndarray): A numpy array containing the input.
        """
        self._mean = np.mean(data, axis=0, keepdims=True)
        self._std = np.std(data, axis=0, keepdims=True)
        self._std = np.maximum(self._std, 1e-12)
        self._mean_t = torch.FloatTensor(self._mean).to(self._device)
        self._std_t = torch.FloatTensor(self._std).to(self._device)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transforms the input matrix data using the parameters of this scaler.

        Args:
            data (torch.Tensor): The input data to transform.

        Returns:
            transformed_data: The transformed data.
        """
        return (data - self._mean_t) / self._std_t


def init_weights(layer: nn.Module) -> None:
    """Initialize network weight.

    Args:
        layer (nn.Module): The layer to initialize.
    """

    def truncated_normal_init(
        weight: torch.Tensor,
        mean: float = 0.0,
        std: float = 0.01,
    ) -> torch.Tensor:
        """Initialize network weight.

        Args:
            weight (torch.Tensor): The weight to be initialized.
            mean (float): The mean of the normal distribution.
            std (float): The standard deviation of the normal distribution.

        Returns:
            weight: The initialized weight.
        """
        torch.nn.init.normal_(weight, mean=mean, std=std)
        while True:
            cond = torch.logical_or(weight < mean - 2 * std, weight > mean + 2 * std)
            if not torch.sum(cond):
                break
            weight = torch.where(
                cond,
                torch.nn.init.normal_(torch.ones(weight.shape), mean=mean, std=std),
                weight,
            )
        return weight

    if isinstance(layer, (nn.Linear, EnsembleFC)):
        input_dim = layer.in_features
        truncated_normal_init(layer.weight, std=1 / (2 * np.sqrt(input_dim)))
        layer.bias.data.fill_(0.0)


def unbatched_forward(
    layer: nn.Module | EnsembleFC,
    input_data: torch.Tensor,
    index: int,
) -> torch.Tensor:
    """Special forward for nn.Sequential modules which contain BatchedLinear layers we want to use.

    Args:
        layer (nn.Module | EnsembleFC): The layer to forward through.
        input_data (torch.Tensor): The input data.
        index (int): The index of the model to use.

    Returns:
        output: The output of the layer.
    """
    if isinstance(layer, EnsembleFC):
        # pylint: disable-next=not-callable
        output = F.linear(
            input_data,
            torch.transpose(layer.weight[index].squeeze(0), 0, 1),
            layer.bias[index],
        )

    else:
        output = layer(input_data)
    return output


class EnsembleFC(nn.Module):
    """Ensemble fully connected network.

    A fully connected network with ensemble_size models.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        ensemble_size (int): The number of models in the ensemble.
        weight_decay (float): The decaying factor.
        bias (bool): Whether to use bias.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        ensemble_size (int): The number of models in the ensemble.
        weight (nn.Parameter): The weight of the network.
        bias (nn.Parameter): The bias of the network.
    """

    _constants_: list[str]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize an instance of fully connected network."""
        super().__init__()
        self._constants_ = ['in_features', 'out_features']
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_data (torch.Tensor): The input data.

        Returns:
            The forward output of the network.
        """
        w_times_x = torch.bmm(input_data, self.weight)
        if self.bias is not None:
            w_times_x = torch.add(w_times_x, self.bias[:, None, :])  # w times x + b
        return w_times_x


# pylint: disable-next=too-many-instance-attributes
class EnsembleModel(nn.Module):
    """Ensemble dynamics model.

    A dynamics model with ensemble_size models.

    Args:
        device (torch.device): The device to use.
        state_size (int): The size of the state.
        action_size (int): The size of the action.
        reward_size (int): The size of the reward.
        cost_size (int): The size of the cost.
        ensemble_size (int): The number of models in the ensemble.
        predict_reward (bool): Whether to predict reward.
        predict_cost (bool): Whether to predict cost.
        hidden_size (int): The size of the hidden layer.
        learning_rate (float): The learning rate.
        use_decay (bool): Whether to use weight decay.

    Attributes:
        max_logvar (torch.Tensor): The maximum log variance.
        min_logvar (torch.Tensor): The minimum log variance.
        scaler (StandardScaler): The scaler.
    """

    max_logvar: torch.Tensor
    min_logvar: torch.Tensor

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        device: torch.device,
        state_size: int,
        action_size: int,
        reward_size: int,
        cost_size: int,
        ensemble_size: int,
        predict_reward: bool,
        predict_cost: bool = False,
        hidden_size: int = 200,
        learning_rate: float = 1e-3,
        use_decay: bool = False,
    ) -> None:
        """Initialize network weight."""
        super().__init__()

        self._state_size: int = state_size
        self._reward_size: int = reward_size
        self._cost_size: int = cost_size
        self._predict_reward: bool = predict_reward
        self._predict_cost: bool = predict_cost

        self._output_dim: int = state_size
        if predict_reward:
            self._output_dim += reward_size
        if predict_cost:
            self._output_dim += cost_size

        self._hidden_size: int = hidden_size
        self._use_decay: bool = use_decay

        self._nn1: EnsembleFC = EnsembleFC(
            state_size + action_size,
            hidden_size,
            ensemble_size,
            weight_decay=0.000025,
        )
        self._nn2: EnsembleFC = EnsembleFC(
            hidden_size,
            hidden_size,
            ensemble_size,
            weight_decay=0.00005,
        )
        self._nn3: EnsembleFC = EnsembleFC(
            hidden_size,
            hidden_size,
            ensemble_size,
            weight_decay=0.000075,
        )
        self._nn4: EnsembleFC = EnsembleFC(
            hidden_size,
            hidden_size,
            ensemble_size,
            weight_decay=0.000075,
        )
        self._nn5: EnsembleFC = EnsembleFC(
            hidden_size,
            self._output_dim * 2,
            ensemble_size,
            weight_decay=0.0001,
        )

        self.register_buffer('max_logvar', (torch.ones((1, self._output_dim)).float() / 2))
        self.register_buffer('min_logvar', (-torch.ones((1, self._output_dim)).float() * 10))
        self._optimizer: torch.optim.Adam = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self._device: torch.device = device
        self.scaler: StandardScaler = StandardScaler(self._device)

    def forward(
        self,
        data: torch.Tensor | np.ndarray,
        ret_log_var: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=not-callable
        """Compute next state, reward, cost using all models.

        Args:
            data (torch.Tensor): Input data.
            ret_log_var (bool, optional): Whether to return the log variance, defaults to False.

        Returns:
            mean: Mean of the next state, reward, cost.
            logvar or var: Log variance of the next state, reward, cost.
        """
        if isinstance(data, torch.Tensor):
            data_t = data
        else:
            data_t = torch.tensor(data, dtype=torch.float32, device=self._device)
        data_t = self.scaler.transform(data_t)
        nn1_output = swish(self._nn1(data_t))
        nn2_output = swish(self._nn2(nn1_output))
        nn3_output = swish(self._nn3(nn2_output))
        nn4_output = swish(self._nn4(nn3_output))
        nn5_output = self._nn5(nn4_output)
        mean = nn5_output[:, :, : self._output_dim]
        logvar = self.max_logvar - F.softplus(
            self.max_logvar - nn5_output[:, :, self._output_dim :],
        )
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)
        if ret_log_var:
            return mean, logvar
        return mean, var

    # pylint: disable=not-callable
    def forward_idx(
        self,
        data: torch.Tensor | np.ndarray,
        idx_model: int,
        ret_log_var: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute next state, reward, cost from an certain model.

        Args:
            data (torch.Tensor | np.ndarray): Input data.
            idx_model (int): Index of the model.
            ret_log_var (bool): Whether to return the log variance.

        Returns:
            mean: Mean of the next state, reward, cost.
            logvar or var: Log variance of the next state, reward, cost.
        """
        assert data.shape[0] == 1
        if isinstance(data, torch.Tensor):
            data_t = data
        else:
            data_t = torch.tensor(data, dtype=torch.float32, device=self._device)
        data_t = self.scaler.transform(data_t[0])
        unbatched_forward_fn = partial(unbatched_forward, index=idx_model)
        nn1_output = swish(unbatched_forward_fn(self._nn1, data_t))
        nn2_output = swish(unbatched_forward_fn(self._nn2, nn1_output))
        nn3_output = swish(unbatched_forward_fn(self._nn3, nn2_output))
        nn4_output = swish(unbatched_forward_fn(self._nn4, nn3_output))
        nn5_output = unbatched_forward_fn(self._nn5, nn4_output)
        mean = nn5_output[:, : self._output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, self._output_dim :])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        if ret_log_var:
            return mean.unsqueeze(0), logvar.unsqueeze(0)
        return mean.unsqueeze(0), var.unsqueeze(0)

    def _get_decay_loss(self) -> torch.Tensor:
        """Get decay loss."""
        decay_loss = torch.tensor(0.0).to(self._device)
        for layer in self.children():
            if isinstance(layer, EnsembleFC):
                decay_loss += layer.weight_decay * torch.sum(torch.square(layer.weight)) / 2.0
        return decay_loss

    def loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
        inc_var_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss.

        Args:
            mean (torch.Tensor): Mean of the next state, reward, cost.
            logvar (torch.Tensor): Log variance of the next state, reward, cost.
            labels (torch.Tensor): Ground truth of the next state, reward, cost.
            inc_var_loss (bool, optional): Whether to include the variance loss. Defaults to True.

        Returns:
            total_loss (torch.Tensor): Total loss.
            mse_loss (torch.Tensor): MSE loss.
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train_ensemble(self, loss: torch.Tensor) -> None:
        """Train the dynamics model.

        Args:
            loss (torch.Tensor): The loss of the dynamics model.
        """
        self._optimizer.zero_grad()
        loss += 0.01 * torch.sum(torch.Tensor(self.max_logvar)) - 0.01 * torch.sum(
            torch.Tensor(self.min_logvar),
        )
        if self._use_decay:
            loss += self._get_decay_loss()
        loss.backward()
        self._optimizer.step()


# pylint: disable-next=too-many-instance-attributes
class EnsembleDynamicsModel:
    """Dynamics model for predict next state, reward and cost.

    Args:
        model_cfgs (Config): The configuration of the dynamics model.
        device (torch.device): The device to use.
        state_shape (tuple[int, ...]): The shape of the state.
        action_shape (tuple[int, ...]): The shape of the action.
        actor_critic (ConstraintActorCritic | ConstraintActorQCritic | None, optional): The actor critic model.
            Defaults to None.
        rew_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The reward function. Defaults to None.
        cost_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The cost function.
            Defaults to None.
        terminal_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The terminal function.
            Defaults to None.

    Attributes:
        elite_model_idxes (list[int]): The index of the elite models.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        model_cfgs: Config,
        device: torch.device,
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = None,
        rew_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cost_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
        terminal_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the dynamics model."""
        self._num_ensemble: int = model_cfgs.num_ensemble
        self._elite_size: int = model_cfgs.elite_size
        self._predict_reward: bool = model_cfgs.predict_reward
        self._predict_cost: bool = model_cfgs.predict_cost
        self._batch_size: int = model_cfgs.batch_size
        self._max_epoch_since_update: int = model_cfgs.max_epoch

        self._reward_size: int = model_cfgs.reward_size
        self._cost_size: int = model_cfgs.cost_size
        self._use_cost: bool = model_cfgs.use_cost
        self._use_terminal: bool = model_cfgs.use_terminal
        self._use_var: bool = model_cfgs.use_var
        self._use_reward_critic: bool = model_cfgs.use_reward_critic
        self._use_cost_critic: bool = model_cfgs.use_cost_critic

        self._device: torch.device = device

        self._state_size: int = state_shape[0]
        self._action_size: int = action_shape[0]

        self._rew_func: Callable[[torch.Tensor], torch.Tensor] | None = rew_func
        self._cost_func: Callable[[torch.Tensor], torch.Tensor] | None = cost_func
        self._terminal_func: Callable[[torch.Tensor], torch.Tensor] | None = terminal_func

        self._actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = actor_critic

        self._model_list: list[int] = []

        self.elite_model_idxes: list[int] = list(range(self._elite_size))
        self._ensemble_model: EnsembleModel = EnsembleModel(
            device=self._device,
            state_size=self._state_size,
            action_size=self._action_size,
            reward_size=self._reward_size,
            cost_size=self._cost_size,
            ensemble_size=self._num_ensemble,
            predict_reward=model_cfgs.predict_reward,
            predict_cost=model_cfgs.predict_cost,
            hidden_size=model_cfgs.hidden_size,
            learning_rate=1e-3,
            use_decay=model_cfgs.use_decay,
        )
        self._ensemble_model.to(self._device)
        self._max_epoch_since_update = 5
        self._epochs_since_update: int = 0
        self._snapshots: dict[int, tuple[int, float]] = {
            i: (0, 1e10) for i in range(self._num_ensemble)
        }

        if self._predict_reward is False:
            assert rew_func is not None, 'rew_func should not be None'
        if self._use_cost is True and self._predict_cost is False:
            assert cost_func is not None, 'cost_func should not be None'
            assert (
                cost_func(torch.zeros((1, self._state_size)).to(self._device)) is not None
            ), 'cost_func should return cost'
        if self._use_terminal is True:
            assert terminal_func is not None, 'terminal_func should not be None'

        self._state_start_dim = (
            int(self._predict_reward) * self._reward_size
            + int(self._predict_cost) * self._cost_size
        )

    @property
    def ensemble_model(self) -> EnsembleModel:
        """The ensemble model."""
        return self._ensemble_model

    @property
    def num_models(self) -> int:
        """The number of ensemble."""
        return self._num_ensemble

    @property
    def state_size(self) -> int:
        """The state size."""
        return self._state_size

    # pylint: disable-next=too-many-locals, too-many-arguments
    def train(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        holdout_ratio: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the dynamics, holdout_ratio is the data ratio hold out for validation.

        Args:
            inputs (np.ndarray): Input data.
            labels (np.ndarray): Ground truth of the next state, reward, cost.
            holdout_ratio (float): The ratio of the data hold out for validation.

        Returns:
            train_mse_losses: The training loss.
            val_mse_losses: The validation loss.
        """
        self._epochs_since_update = 0
        self._snapshots = {i: (0, 1e10) for i in range(self._num_ensemble)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        # split training and testing dataset
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        self._ensemble_model.scaler.fit(train_inputs)

        train_mse_losses = []
        val_losses = []
        for epoch in itertools.count():
            # training
            train_idx = np.vstack(
                [np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_ensemble)],
            )
            # shape: [train_inputs.shape[0],num_ensemble]

            for start_pos in range(0, train_inputs.shape[0], self._batch_size):
                idx = train_idx[:, start_pos : start_pos + self._batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self._device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self._device)
                mean, logvar = self._ensemble_model.forward(train_input, ret_log_var=True)
                total_loss, mse_loss = self._ensemble_model.loss(mean, logvar, train_label)
                self._ensemble_model.train_ensemble(total_loss)
                train_mse_losses.append(mse_loss.detach().cpu().numpy().mean())

            # validation
            val_idx = np.vstack(
                [np.random.permutation(holdout_inputs.shape[0]) for _ in range(self._num_ensemble)],
            )
            val_batch_size = 512
            val_losses_list = []
            for start_pos in range(0, holdout_inputs.shape[0], val_batch_size):
                with torch.no_grad():
                    idx = val_idx[:, start_pos : start_pos + val_batch_size]
                    val_input = torch.from_numpy(holdout_inputs[idx]).float().to(self._device)
                    val_label = torch.from_numpy(holdout_labels[idx]).float().to(self._device)
                    holdout_mean, holdout_logvar = self._ensemble_model.forward(
                        val_input,
                        ret_log_var=True,
                    )
                    _, holdout_mse_losses = self._ensemble_model.loss(
                        holdout_mean,
                        holdout_logvar,
                        val_label,
                        inc_var_loss=False,
                    )
                    holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                    val_losses_list.append(holdout_mse_losses)
            current_loss = np.sum(np.array(val_losses_list), axis=0) / (
                int(holdout_inputs.shape[0] / val_batch_size) + 1
            )
            val_losses.append(current_loss)
            sorted_loss_idx = np.argsort(current_loss)
            self.elite_model_idxes = sorted_loss_idx[: self._elite_size].tolist()
            break_train = self._save_best(epoch, current_loss)
            if break_train:
                break

        train_mse_losses = np.array(train_mse_losses).mean()
        val_mse_losses = np.array(val_losses).mean()
        return np.array(train_mse_losses), np.array(val_mse_losses)

    def _save_best(self, epoch: int, holdout_losses: list) -> bool:
        """Save the best model.

        Args:
            epoch (int): The current epoch.
            holdout_losses (list): The holdout loss.

        Returns:
            Whether to break the training.
        """
        updated = False
        for i, current_loss in enumerate(holdout_losses):
            _, best = self._snapshots[i]
            improvement = (best - current_loss) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current_loss)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        return self._epochs_since_update > self._max_epoch_since_update

    @torch.no_grad()
    def _compute_reward(
        self,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reward from the network output.

        Args:
            network_output (torch.Tensor): The output of the network.

        Returns:
            reward: The reward, from the network output or the reward function.

        Raises:
            ValueError: If the reward function is not defined.
        """
        if self._predict_reward:
            reward_start_dim = int(self._predict_cost) * self._cost_size
            reward_end_dim = reward_start_dim + self._reward_size
            return network_output[:, :, reward_start_dim:reward_end_dim]
        if self._rew_func is not None:
            return self._rew_func(network_output[:, :, self._state_start_dim :])
        raise ValueError('Reward function is not defined.')

    @torch.no_grad()
    def _compute_cost(
        self,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cost from the network output.

        Args:
            network_output (torch.Tensor): The output of the network.

        Returns:
            cost: The cost, from the network output or the cost function.

        Raises:
            ValueError: If the cost function is not defined.
        """
        if self._predict_cost:
            return network_output[:, :, : self._cost_size]
        if self._cost_func is not None:
            return self._cost_func(network_output[:, :, self._state_start_dim :])
        raise ValueError('Cost function is not defined.')

    @torch.no_grad()
    def _compute_terminal(
        self,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the terminal from the network output.

        Args:
            network_output (torch.Tensor): The output of the network.

        Returns:
            terminal: The terminal signal, from the network output or the terminal function.

        Raises:
            ValueError: If the terminal function is not defined.
        """
        if self._terminal_func is not None:
            return self._terminal_func(network_output[:, :, self._state_start_dim :])
        raise ValueError('Terminal function is not defined.')

    def _predict(
        self,
        inputs: torch.Tensor,
        batch_size: int = 1024,
        idx: int | None = None,
        ret_log_var: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input type and output type both are tensor, used for planning loop.

        Args:
            inputs (torch.Tensor): the inputs to the network.
            batch_size (int, optional): the batch size for prediction.
            idx (Union[int, None], optional): the index of the model to use.
            ret_log_var (bool, optional): whether to return the log variance.

        Returns:
            ensemble_mean_tensor: The mean of the ensemble.
            ensemble_var_tensor: The variance of the ensemble.
        """
        if idx is not None:
            assert inputs.shape[0] == 1
        else:
            assert inputs.shape[0] == self._num_ensemble
        # input shape: [networ_size, (num_gaus+num_actor)*paritcle ,state_dim + action_dim]
        assert inputs.shape[2] == self._state_size + self._action_size

        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[1], batch_size):
            model_input = inputs[:, i : min(i + batch_size, inputs.shape[1]), :]
            # input shape: [networ_size, (num_gaus+num_actor)*paritcle ,state_dim + action_dim]
            if idx is None:
                b_mean, b_var = self._ensemble_model.forward(
                    model_input,
                    ret_log_var=ret_log_var,
                )
            else:
                b_mean, b_var = self._ensemble_model.forward_idx(
                    model_input,
                    idx,
                    ret_log_var=ret_log_var,
                )
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean_tensor = torch.cat(ensemble_mean, dim=1)
        ensemble_var_tensor = torch.cat(ensemble_var, dim=1)
        assert (
            ensemble_mean_tensor.shape[:-1] == inputs.shape[:-1]
            and ensemble_var_tensor.shape[:-1] == inputs.shape[:-1]
        ), 'output shape must be the same as input shape except the last dimension'
        return ensemble_mean_tensor, ensemble_var_tensor

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        idx: int | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
        # pylint: disable-next=line-too-long
        """Sample states and rewards from the ensemble model.

        Args:
            states (torch.Tensor): the states.
            actions (torch.Tensor): the actions.
            idx (Union[int, None], optional): the index of the model to use. Defaults to None.
            deterministic (bool, optional): whether to use the deterministic version of the model. Defaults to False.

        Returns:
            sample_states (torch.Tensor): the sampled states.
            rewards (torch.Tensor): the rewards.
            info: the info dict, contains the costs if `use_cost` is True.
        """
        assert (
            states.shape[:-1] == actions.shape[:-1]
        ), 'states and actions must have the same shape except the last dimension'
        # shape: [network_size, (num_gaus+num_actor)*paritcle ,state_dim]
        inputs = torch.cat((states, actions), dim=-1)

        ensemble_mean, ensemble_var = self._predict(inputs, idx=idx)
        ensemble_mean[:, :, self._state_start_dim :] += states
        ensemble_std = torch.sqrt(ensemble_var)
        if deterministic:
            ensemble_samples = ensemble_mean
        else:
            ensemble_samples = (
                ensemble_mean
                + torch.randn(size=ensemble_mean.shape).to(self._device) * ensemble_std
            )
        states = ensemble_samples[:, :, self._state_start_dim :]
        # shape: [network_size, (num_gaus+num_actor)*paritcle ,state_dim]
        rewards = self._compute_reward(ensemble_samples)
        # shape: [network_size, (num_gaus+num_actor)*paritcle ,reward_dim]
        info = defaultdict(list)
        if self._use_cost:
            info['costs'].append(self._compute_cost(ensemble_samples))
        if self._use_terminal:
            info['terminals'].append(self._compute_terminal(ensemble_samples))
        if self._use_var:
            info['vars'].append(ensemble_var)
        if (
            self._use_reward_critic
            and self._actor_critic is not None
            and hasattr(self._actor_critic, 'reward_critic')
        ):
            reward_values = self._actor_critic.reward_critic(
                states.reshape(-1, self._state_size),
                actions.reshape(-1, self._action_size),
            )[0]
            info['values'].append(reward_values.reshape((*states.shape[:-1], 1)))
        if (
            self._use_cost_critic
            and self._actor_critic is not None
            and hasattr(self._actor_critic, 'cost_critic')
        ):
            cost_values = self._actor_critic.cost_critic(
                states.reshape(-1, self._state_size),
                actions.reshape(-1, self._action_size),
            )
            info['cost_values'].append(cost_values.reshape((*states.shape[:-1], 1)))
        return states, rewards, info

    @torch.no_grad()
    def imagine(
        self,
        states: torch.Tensor,
        horizon: int,
        actions: torch.Tensor | None = None,
        actor_critic: ConstraintActorQCritic | None = None,
        idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Imagine the future states and rewards from the ensemble model.

        Args:
            states (torch.Tensor): the states.
            horizon (int): the horizon.
            actions (torch.Tensor, optional): the actions.
            actor_critic (ConstraintActorQCritic, optional): the actor_critic to use if actions is None.
            idx (int, optional): the index of the model to use.

        Returns:
            traj: the trajectory dict, contains the states, rewards, etc.
        """
        assert (
            states.shape[1] == self._state_size
        ), 'states should be of shape (batch_size, state_size)'
        num_ensemble = self._num_ensemble if idx is None else 1
        if actions is not None:
            assert actions.shape == torch.Size(
                [horizon, states.shape[0], self._action_size],
            ), 'actions should be of shape (horizon, batch_size, action_size)'
            actions = actions[:, None, :, :].repeat([1, num_ensemble, 1, 1])

            # pylint: disable-next=unused-argument
            def get_action(state: torch.Tensor, step: int) -> torch.Tensor:
                assert actions is not None
                return actions[step]

        else:
            # pylint: disable-next=unused-argument
            def get_action(state: torch.Tensor, step: int) -> torch.Tensor:
                assert actor_critic is not None and hasattr(
                    actor_critic,
                    'actor',
                ), 'actor_critic must have an actor'
                return actor_critic.actor.predict(state, deterministic=False)

        traj = defaultdict(list)
        states = states[None, :, :].repeat([num_ensemble, 1, 1])

        for step in range(horizon):
            actions_t = get_action(states, step)
            states, rewards, info = self.sample(states, actions_t, idx)
            states = torch.clamp(torch.nan_to_num(states, nan=0, posinf=0, neginf=0), -100, 100)
            rewards = torch.nan_to_num(rewards, nan=0, posinf=0, neginf=0)

            traj['states'].append(states)
            traj['actions'].append(actions_t)
            traj['rewards'].append(rewards)
            for key, value in info.items():
                value_ = torch.nan_to_num(value[0], nan=0, posinf=0, neginf=0)
                traj[key].append(value_.clone())
        traj_tensor = {}
        for key, value in traj.items():
            traj_tensor[key] = torch.stack(value, dim=0)
        return traj_tensor
