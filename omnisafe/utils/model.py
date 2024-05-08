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
"""This module contains the helper functions for the model."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from omnisafe.typing import DEVICE_CPU, Activation, InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')


def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from: ``identity``, ``relu``, ``sigmoid``, ``softplus``,
    ``tanh``.

    Args:
        activation (Activation): The activation function.

    Returns:
        The activation function, ranging from ``nn.Identity``, ``nn.ReLU``, ``nn.Sigmoid``,
        ``nn.Softplus`` to ``nn.Tanh``.
    """
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
    }
    assert activation in activations
    return activations[activation]


def build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn()]
    return nn.Sequential(*layers)


class ObservationConcator:
    """A class designed to concatenate observations and actions over a specified time steps."""

    def __init__(
        self,
        state_shape: tuple,
        action_shape: tuple,
        num_sequences: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize the ObservationConcator with given shapes and device.

        Args:
            state_shape (tuple): Shape of the state space.
            action_shape (tuple): Shape of the action space.
            num_sequences (int): Number of sequences to maintain in the history.
            device (str): The device (CPU/GPU) on which to create tensors.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences
        self.device = device
        self._state: deque = deque(maxlen=self.num_sequences)
        self._action: deque = deque(maxlen=self.num_sequences - 1)

    def reset_episode(self, state: torch.Tensor) -> None:
        """Reset the history of states and actions for a new episode.

        Args:
            state (torch.Tensor): The initial state for the new episode.
        """
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(
                torch.zeros(self.state_shape, dtype=torch.float32, device=self.device),
            )
            self._action.append(
                torch.zeros(self.action_shape, dtype=torch.float32, device=self.device),
            )
        self._state.append(state)

    def append(self, state: torch.Tensor, action: torch.Tensor) -> None:
        """Append a new state and action to the queue.

        Args:
            state (torch.Tensor): State to be appended.
            action (torch.Tensor): Action to be appended.
        """
        self._state.append(state)
        self._action.append(action)

    @property
    def last_state(self) -> torch.Tensor:
        """Returns the most recent state.

        Returns:
            torch.Tensor: The most recent state.
        """
        return self._state[-1][None, ...]

    @property
    def last_action(self) -> torch.Tensor:
        """Returns the most recent action.

        Returns:
            torch.Tensor: The most recent action.
        """
        return self._action[-1]
