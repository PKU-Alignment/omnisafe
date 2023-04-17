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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


def swish(data):
    """Transform data using sigmoid function."""
    return data * torch.sigmoid(data)


class StandardScaler:
    """Normalize data"""

    def __init__(self, device: str) -> None:
        self._mean = 0.0
        self._std = 1.0
        self._mean_t = torch.tensor(self._mean).to(device)
        self._std_t = torch.tensor(self._std).to(device)
        self._device = device

    def fit(self, data: torch.Tensor | np.ndarray) -> None:
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Args:
            data (np.ndarray): A numpy array containing the input
        """
        self._mean = np.mean(data, axis=0, keepdims=True)
        self._std = np.std(data, axis=0, keepdims=True)
        self._std[self._std < 1e-12] = 1.0
        self._mean_t = torch.FloatTensor(self._mean).to(self._device)
        self._std_t = torch.FloatTensor(self._std).to(self._device)

    def transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
            data (torch.Tensor|np.ndarray): A numpy array containing the input

        Returns:
            transformed_data (torch.Tensor|np.ndarray): The transformed data.
        """
        if torch.is_tensor(data):
            return (data - self._mean_t) / self._std_t
        return (data - self._mean) / self._std


def init_weights(layer: nn.Module) -> None:
    """Initialize network weight"""

    def truncated_normal_init(weight, mean: float = 0.0, std: float = 0.01) -> torch.Tensor:
        """Initialize network weight"""
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
    """Special forward for nn.Sequential modules which contain BatchedLinear layers,
    for when we only want to use one of the models.

    Args:
        layer (nn.Module|EnsembleFC): The layer to forward through.
        input_data (torch.Tensor): The input data.
        index (int): The index of the model to use.

    Returns:
        output (torch.Tensor): The output of the layer.
    """
    if isinstance(layer, EnsembleFC):
        output = F.linear(
            input_data,
            torch.transpose(layer.weight[index].squeeze(0), 0, 1),
            layer.bias[index],
        )

    else:
        output = layer(input_data)
    return output


class EnsembleFC(nn.Module):
    """Ensemble fully connected network."""

    _constants_ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
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
        """Forward pass."""
        w_times_x = torch.bmm(input_data, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b


# pylint: disable-next=too-many-instance-attributes
class EnsembleModel(nn.Module):
    """Ensemble dynamics model."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        state_size,
        action_size,
        reward_size,
        cost_size,
        ensemble_size,
        predict_reward,
        predict_cost=None,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device='cpu',
    ) -> None:
        super().__init__()

        self._state_size = state_size
        self._reward_size = reward_size
        self._cost_size = cost_size
        self._predict_reward = predict_reward
        self._predict_cost = predict_cost

        self._output_dim = state_size
        if predict_reward:
            self._output_dim += reward_size
        if predict_cost:
            self._output_dim += cost_size

        self._hidden_size = hidden_size
        self._use_decay = use_decay

        self._nn1 = EnsembleFC(
            state_size + action_size,
            hidden_size,
            ensemble_size,
            weight_decay=0.000025,
        )
        self._nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self._nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self._nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self._nn5 = EnsembleFC(
            hidden_size,
            self._output_dim * 2,
            ensemble_size,
            weight_decay=0.0001,
        )

        self.register_buffer('max_logvar', (torch.ones((1, self._output_dim)).float() / 2))
        self.register_buffer('min_logvar', (-torch.ones((1, self._output_dim)).float() * 10))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self._device = device
        self.scaler = StandardScaler(self._device)

    # pylint: disable-next=too-many-locals
    def forward(
        self,
        data: torch.Tensor,
        ret_log_var: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute next state, reward, cost using all models.

        Args:
            data (torch.Tensor): Input data.
            ret_log_var (bool): Whether to return the log variance.

        Returns:
            mean (torch.Tensor): Mean of the next state, reward, cost.
            logvar or var (torch.Tensor): Log variance of the next state, reward, cost.
        """
        data = self.scaler.transform(data)
        nn1_output = swish(self._nn1(data))
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

    def forward_idx(
        self,
        data: torch.Tensor,
        idx_model: int,
        ret_log_var: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute next state, reward, cost from an certain model.

        Args:
            data (torch.Tensor): Input data.
            idx_model (int): Index of the model.
            ret_log_var (bool): Whether to return the log variance.

        Returns:
            mean (torch.Tensor): Mean of the next state, reward, cost.
            logvar or var (torch.Tensor): Log variance of the next state, reward, cost.
        """
        assert data.shape[0] == 1
        data = self.scaler.transform(data[0])
        unbatched_forward_fn = partial(unbatched_forward, index=idx_model)
        nn1_output = swish(unbatched_forward_fn(self._nn1, data))
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

    def _get_decay_loss(self):
        """Get decay loss."""
        decay_loss = 0.0
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
            inc_var_loss (bool): Whether to include the variance loss.

        Returns:
            total_loss (torch.Tensor): Total loss.
            mse_loss (torch.Tensor): MSE loss.
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train_ensemble(self, loss):
        """Train the dynamics model."""
        self._optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self._use_decay:
            loss += self._get_decay_loss()
        loss.backward()
        self._optimizer.step()


# pylint: disable-next=too-many-instance-attributes
class EnsembleDynamicsModel:
    """Dynamics model for predict next state, reward and cost."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        model_cfgs,
        device,
        state_size,
        action_size,
        reward_size,
        cost_size,
        use_cost,
        use_terminal,
        use_var,
        use_reward_critic,
        use_cost_critic,
        actor_critic=None,
        rew_func=None,
        cost_func=None,
        terminal_func=None,
    ) -> None:
        self._num_ensemble = model_cfgs.num_ensemble
        self._elite_size = model_cfgs.elite_size
        self._predict_reward = model_cfgs.predict_reward
        self._predict_cost = model_cfgs.predict_cost
        self._batch_size = model_cfgs.batch_size
        self._max_epoch_since_update = model_cfgs.max_epoch
        self._model_list = []
        self._state_size = state_size
        self._action_size = action_size
        self._reward_size = reward_size
        self._cost_size = cost_size
        self._device = device
        self.elite_model_idxes = list(range(self._elite_size))
        self._ensemble_model = EnsembleModel(
            state_size=state_size,
            action_size=action_size,
            reward_size=reward_size,
            cost_size=cost_size,
            ensemble_size=model_cfgs.num_ensemble,
            predict_reward=model_cfgs.predict_reward,
            predict_cost=model_cfgs.predict_cost,
            hidden_size=model_cfgs.hidden_size,
            learning_rate=1e-3,
            use_decay=model_cfgs.use_decay,
            device=self._device,
        )

        self._ensemble_model.to(self._device)
        self._max_epoch_since_update = 5
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self._num_ensemble)}

        if self._predict_reward is False:
            assert rew_func is not None, 'rew_func should not be None'
        if use_cost is True and self._predict_cost is False:
            assert cost_func is not None, 'cost_func should not be None'
        if use_terminal is True:
            assert terminal_func is not None, 'terminal_func should not be None'
        self._rew_func = rew_func
        self._cost_func = cost_func
        self._terminal_func = terminal_func
        self._state_start_dim = (
            int(self._predict_reward) * self._reward_size
            + int(self._predict_cost) * self._cost_size
        )
        self._use_cost = use_cost
        self._use_terminal = use_terminal
        self._use_var = use_var
        self._use_reward_critic = use_reward_critic
        self._use_cost_critic = use_cost_critic
        self._actor_critic = actor_critic

    # pylint: disable-next=too-many-locals, too-many-arguments
    def train(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        holdout_ratio: float = 0.0,
    ) -> None:
        """Train the dynamics, holdout_ratio is the data ratio hold out for validation.

        Args:
            inputs (np.ndarray): Input data.
            labels (np.ndarray): Ground truth of the next state, reward, cost.
            holdout_ratio (float): The ratio of the data hold out for validation.
        """
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self._num_ensemble)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        # split training and testing dataset
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        self._ensemble_model.scaler.fit(train_inputs)

        for epoch in itertools.count():
            train_mse_losses = []
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
            len_valid = 0
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
                len_valid += 1
            val_losses = np.array(val_losses_list)
            val_losses = np.sum(val_losses, axis=0) / len_valid
            sorted_loss_idx = np.argsort(val_losses)
            self.elite_model_idxes = sorted_loss_idx[: self._elite_size].tolist()
            break_train = self._save_best(epoch, val_losses)
            if break_train:
                break

        train_mse_losses = np.array(train_mse_losses).mean()
        val_mse_losses = np.array(val_losses).mean()
        return train_mse_losses, val_mse_losses

    def _save_best(self, epoch: int, holdout_losses: np.ndarray) -> bool:
        """Save the best model."""
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
        """Compute the reward from the network output."""
        if self._predict_reward:
            reward_start_dim = int(self._predict_cost) * self._cost_size
            reward_end_dim = reward_start_dim + self._reward_size
            return network_output[:, :, reward_start_dim:reward_end_dim]
        return self._rew_func(network_output[:, :, self._state_start_dim :])

    @torch.no_grad()
    def _compute_cost(
        self,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cost from the network output."""
        if self._predict_cost:
            return network_output[:, :, : self._cost_size]
        return self._cost_func(network_output[:, :, self._state_start_dim :])

    @torch.no_grad()
    def _compute_terminal(
        self,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the terminal from the network output."""
        return self._terminal_func(network_output[:, :, self._state_start_dim :])

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
        ensemble_mean = torch.cat(ensemble_mean, dim=1)
        ensemble_var = torch.cat(ensemble_var, dim=1)
        assert (
            ensemble_mean.shape[:-1] == inputs.shape[:-1]
            and ensemble_var.shape[:-1] == inputs.shape[:-1]
        ), 'output shape must be the same as input shape except the last dimension'
        return ensemble_mean, ensemble_var

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
            info (Dict[str, List[torch.Tensor]]): the info dict, contains the costs if use_cost is True
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
        if self._use_reward_critic:
            reward_values = self._actor_critic.reward_critic(
                states.reshape(-1, self._state_size),
                actions.reshape(-1, self._action_size),
            )[0]
            info['values'].append(reward_values.reshape((*states.shape[:-1], 1)))
        if self._use_cost_critic:
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
        actor_critic: ConstraintActorQCritic = None,
        idx: int | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Imagine the future states and rewards from the ensemble model.

        Args:
            states (torch.Tensor): the states.
            horizon (int): the horizon.
            actions (torch.Tensor, optional): the actions.
            actor_critic (ConstraintActorQCritic, optional): the actor_critic to use if actions is None.
            idx (int, optional): the index of the model to use.

        Returns:
            traj (Dict[str, List[torch.Tensor]]): the trajectory dict, contains the states, rewards, etc.
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

        else:
            assert actor_critic is not None, 'Need to provide actions or actor_critic'

        traj = defaultdict(list)
        states = states[None, :, :].repeat([num_ensemble, 1, 1])

        for step in range(horizon):
            actions_t = (
                actions[step]
                if actions is not None
                else actor_critic.actor.predict(states, deterministic=False)
            )
            states, rewards, info = self.sample(states, actions_t, idx)
            states = torch.clamp(torch.nan_to_num(states, nan=0, posinf=0, neginf=0), -100, 100)
            rewards = torch.nan_to_num(rewards, nan=0, posinf=0, neginf=0)

            traj['states'].append(states)
            traj['actions'].append(actions_t)
            traj['rewards'].append(rewards)
            for key, value in info.items():
                value_ = torch.nan_to_num(value[0], nan=0, posinf=0, neginf=0)
                traj[key].append(value_.clone())
        for key, value in traj.items():
            traj[key] = torch.stack(value, dim=0)
        return traj
