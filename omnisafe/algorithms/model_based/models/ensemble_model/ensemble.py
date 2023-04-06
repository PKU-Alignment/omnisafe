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
# Modified version of model.py from  https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py
# original version doesn't validate model error batch-wise and is highly memory intensive.
# ==============================================================================
"""Dynamics Model"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import defaultdict
from typing import Dict, List, Tuple, Union

def swish(data):
    """Transform data using sigmoid function."""
    return data * torch.sigmoid(data)


class StandardScaler:
    """Normalize data"""

    def __init__(self, device=torch.device('cpu')):
        self._mean = 0.0
        self._std = 1.0
        self._mean_t = torch.tensor(self._mean).to(device)
        self._std_t = torch.tensor(self._std).to(device)
        self._device = device

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self._mean = np.mean(data, axis=0, keepdims=True)
        self._std = np.std(data, axis=0, keepdims=True)
        self._std[self._std < 1e-12] = 1.0
        self._mean_t = torch.FloatTensor(self._mean).to(self._device)
        self._std_t = torch.FloatTensor(self._std).to(self._device)

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if torch.is_tensor(data):
            return (data - self._mean_t) / self._std_t
        return (data - self._mean) / self._std


def init_weights(layer):
    """Initialize network weight"""

    def truncated_normal_init(weight, mean=0.0, std=0.01):
        """Initialize network weight"""
        torch.nn.init.normal_(weight, mean=mean, std=std)
        while True:
            cond = torch.logical_or(weight < mean - 2 * std, weight > mean + 2 * std)
            if not torch.sum(cond):
                break
            weight = torch.where(
                cond, torch.nn.init.normal_(torch.ones(weight.shape), mean=mean, std=std), weight
            )
        return weight

    if isinstance(layer, (nn.Linear, EnsembleFC)):
        input_dim = layer.in_features
        truncated_normal_init(layer.weight, std=1 / (2 * np.sqrt(input_dim)))
        layer.bias.data.fill_(0.0)

# Special forward for nn.Sequential modules which contain BatchedLinear layers,
# for when we only want to use one of the models.
def unbatched_forward(layer, input, index):
    if isinstance(layer, EnsembleFC):
        weight = layer.weight[index]
        w_times_x = torch.bmm(input.float(), weight)
        return torch.add(w_times_x, layer.bias[index, None, :])  # w times x + b
    else:
        input = layer(input)
        return input


class EnsembleFC(nn.Module):
    """Ensemble fully connected network"""

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
        """forward"""
        w_times_x = torch.bmm(input_data, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b


# pylint: disable-next=too-many-instance-attributes
class EnsembleModel(nn.Module):
    """Ensemble dynamics model"""

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
        device=torch.device('cpu'),
    ):
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
            state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025
        )
        self._nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self._nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self._nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self._nn5 = EnsembleFC(hidden_size, self._output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.register_buffer('max_logvar', (torch.ones((1, self._output_dim)).float() / 2))
        self.register_buffer('min_logvar', (-torch.ones((1, self._output_dim)).float() * 10))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self._device = device
        self.scaler = StandardScaler(self._device)

    # pylint: disable-next=too-many-locals
    def forward_all(self, data, ret_log_var=False):
        """Compute next state, reward, cost"""
        data = self.scaler.transform(data)
        nn1_output = swish(self._nn1(data))
        nn2_output = swish(self._nn2(nn1_output))
        nn3_output = swish(self._nn3(nn2_output))
        nn4_output = swish(self._nn4(nn3_output))
        nn5_output = self._nn5(nn4_output)
        mean = nn5_output[:, :, : self._output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self._output_dim :])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)
        if ret_log_var:
            return mean, logvar
        return mean, var

    def forward_idx(self, data, idx_model, ret_log_var=False):
        assert data.shape[0] == 1
        data = self.scaler.transform(data)
        unbatched_forward_fn = partial(unbatched_forward, index=idx_model)
        nn1_output = swish(unbatched_forward_fn(self._nn1, data))
        nn2_output = swish(unbatched_forward_fn(self._nn2, nn1_output))
        nn3_output = swish(unbatched_forward_fn(self._nn3, nn2_output))
        nn4_output = swish(unbatched_forward_fn(self._nn4, nn3_output))
        nn5_output = unbatched_forward_fn(self._nn5, nn4_output)
        mean = nn5_output[:, :, : self._output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self._output_dim :])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        if ret_log_var:
            return mean, logvar
        return mean, var

    def _get_decay_loss(self):
        """Get decay loss"""
        decay_loss = 0.0
        for layer in self.children():
            if isinstance(layer, EnsembleFC):
                decay_loss += layer.weight_decay * torch.sum(torch.square(layer.weight)) / 2.0
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
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
        """Train the dynamics model"""
        self._optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self._use_decay:
            loss += self._get_decay_loss()
        loss.backward()
        self._optimizer.step()


# pylint: disable-next=too-many-instance-attributes
class EnsembleDynamicsModel:
    """Dynamics model for predict next state, reward and cost"""

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
        use_truncated,
        use_var,
        use_reward_critic,
        use_cost_critic,
        actor_critic=None,
        rew_func=None,
        cost_func=None,
        truncated_func=None,
    ):
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
            device=self._device
        )

        self._ensemble_model.to(self._device)
        self._max_epoch_since_update = 5
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self._num_ensemble)}

        if self._predict_reward is False:
            assert rew_func is not None, "rew_func should not be None"
        if use_cost is True:
            if self._predict_cost is False:
                assert cost_func is not None, "cost_func should not be None"
        if use_truncated is True:
            assert truncated_func is not None, "truncated_func should not be None"
        self._rew_func = rew_func
        self._cost_func = cost_func
        self._truncated_func = truncated_func
        self._state_start_dim = int(self._predict_reward)*self._reward_size + int(self._predict_cost)*self._cost_size
        self._use_cost = use_cost
        self._use_truncated = use_truncated
        self._use_var = use_var
        self._use_reward_critic = use_reward_critic
        self._use_cost_critic = use_cost_critic
        self._actor_critic = actor_critic

    # pylint: disable-next=too-many-locals, too-many-arguments
    def train(self, inputs, labels, holdout_ratio=0.0):
        """train dynamics, holdout_ratio is the data ratio hold out for validation"""
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
        #train_inputs = self._ensemble_model.scaler.transform(train_inputs)
        #holdout_inputs = self._ensemble_model.scaler.transform(holdout_inputs)

        for epoch in itertools.count():
            train_mse_losses = []
            # training
            train_idx = np.vstack(
                [np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_ensemble)]
            )
            # shape: [train_inputs.shape[0],num_ensemble]

            for start_pos in range(0, train_inputs.shape[0], self._batch_size):
                idx = train_idx[:, start_pos : start_pos + self._batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self._device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self._device)
                mean, logvar = self._ensemble_model.forward_all(train_input, ret_log_var=True)
                total_loss, mse_loss = self._ensemble_model.loss(mean, logvar, train_label)
                self._ensemble_model.train_ensemble(total_loss)
                train_mse_losses.append(mse_loss.detach().cpu().numpy().mean())

            # validation
            val_idx = np.vstack(
                [np.random.permutation(holdout_inputs.shape[0]) for _ in range(self._num_ensemble)]
            )
            val_batch_size = 512
            val_losses_list = []
            len_valid = 0
            for start_pos in range(0, holdout_inputs.shape[0], val_batch_size):
                with torch.no_grad():
                    idx = val_idx[:, start_pos : start_pos + val_batch_size]
                    val_input = torch.from_numpy(holdout_inputs[idx]).float().to(self._device)
                    val_label = torch.from_numpy(holdout_labels[idx]).float().to(self._device)
                    holdout_mean, holdout_logvar = self._ensemble_model.forward_all(val_input, ret_log_var=True)
                    _, holdout_mse_losses = self._ensemble_model.loss(
                        holdout_mean, holdout_logvar, val_label, inc_var_loss=False
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

    def _save_best(self, epoch, holdout_losses):
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

    def _compute_reward(
            self,
            network_output: torch.Tensor,
            ) -> torch.Tensor:
        if self._predict_reward:
            return network_output[:,:,:self._reward_size]
        else:
            return self._rew_func(network_output[:,:,self._state_start_dim:])

    def _compute_cost(
            self,
            network_output: torch.Tensor,
            ) -> torch.Tensor:
        if self._predict_cost:
            cost_start_dim = int(self._predict_reward)*self._reward_size
            cost_end_dim = cost_start_dim + self._cost_size
            return network_output[:,:,cost_start_dim:cost_end_dim]
        else:
            return self._cost_func(network_output[:,:,self._state_start_dim:])

    def _compute_truncated(
            self,
            network_output: torch.Tensor,
            ) -> torch.Tensor:
        return self._trunc_func(network_output[:,:,self._state_start_dim:])

    def _predict(
            self,
            inputs: torch.Tensor,
            batch_size: int=1024,
            idx: Union[int, None]=None,
            ret_log_var: bool=False
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input type and output type both are tensor, used for planning loop"""
        # input shape: [networ_size, (num_gaus+num_actor)*paritcle ,state_dim + action_dim]
        if idx is not None:
            assert inputs.shape[0] == 1
        else:
            assert inputs.shape[0] == self._num_ensemble
        assert inputs.shape[2] == self._state_size + self._action_size

        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[1], batch_size):
            model_input = inputs[:,i : min(i + batch_size, inputs.shape[1]),:]
            # input shape: [networ_size, (num_gaus+num_actor)*paritcle ,state_dim + action_dim]
            if idx is None:
                b_mean, b_var = self._ensemble_model.forward_all(model_input, ret_log_var=ret_log_var)
            else:
                b_mean, b_var = self._ensemble_model.forward_idx(model_input, idx, ret_log_var=ret_log_var)
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean = torch.cat(ensemble_mean, dim=1)
        ensemble_var = torch.cat(ensemble_var, dim=1)
        assert ensemble_mean.shape[:-1] == inputs.shape[:-1] and ensemble_var.shape[:-1] == inputs.shape[:-1], "output shape must be the same as input shape except the last dimension"
        return ensemble_mean, ensemble_var


    def sample(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            idx: Union[int, None]=None,
            deterministic: bool=False,
            ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[torch.Tensor]]]:

        assert states.shape[:-1] == actions.shape[:-1], "states and actions must have the same shape except the last dimension"


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
        rewards = self._compute_reward(ensemble_samples)
        info = defaultdict(list)
        if self._use_cost:
            info['costs'].append(self._compute_cost(ensemble_samples))
        if self._use_truncated:
            info['truncated'].append(self._compute_truncated(ensemble_samples))
        if self._use_var:
            info['var'].append(ensemble_var)
        if self._use_reward_critic:
            reward_values = self._actor_critic.reward_critic(states.reshape(-1, self._state_size), actions.reshape(-1, self._action_size)).reshape((states.shapes[:-1],1))
            info['value'].append(reward_values)
        if self._use_cost_critic:
            cost_values = self._actor_critic.cost_critic(states.reshape(-1, self._state_size), actions.reshape(-1, self._action_size)).reshape((states.shapes[:-1],1))
            info['cost_value'].append(cost_values)
        return states, rewards, info

    def imagine(
            self,
            states: torch.Tensor,
            horizon: int,
            actions: Union[torch.Tensor, None]=None,
            idx: Union[int, None]=None,
            ) -> Dict[str, List[torch.Tensor]]:

        assert states.shape[1] == self._state_size, "states should be of shape (batch_size, state_size)"
        if actions is not None:
            assert actions.shape == torch.Size([horizon, states.shape[0], self._action_size]), "actions should be of shape (horizon, batch_size, action_size)"
        else:
            assert self._actor_critic is not None, "Need to provide actions or actor_critic"

        if idx is None:
            num_ensemble = self._num_ensemble
        else:
            num_ensemble = 1

        traj = defaultdict(list)
        states = states[None, :, :].repeat([num_ensemble, 1, 1])

        for step in range(horizon):
            actions_t = actions[step] if actions is not None else self._actor_critic.actor.predict(states, deterministic=False)
            actions_t = actions_t[None, :, :].repeat([num_ensemble, 1, 1])
            states, rewards, info = self.sample(states, actions_t, idx)


            traj['states'].append(states)
            traj['actions'].append(actions_t)
            traj['rewards'].append(rewards)
            for key, value in info.items():
                traj[key].append(value)
        for key, value in traj.items():
            traj[key] = (torch.stack(value, dim=0))
        return traj