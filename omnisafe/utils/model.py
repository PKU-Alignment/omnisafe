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
"""This module contains the helper functions for the model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from omnisafe.typing import Activation, InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from:
    ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
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
) -> type[nn.Identity] | type[nn.ReLU] | type[nn.Sigmoid] | type[nn.Softplus] | type[nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from:
    ``identity``, ``relu``, ``sigmoid``, ``softplus``, ``tanh``.

    Args:
        activation (Activation): The activation function.
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

    Example:
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
        sizes (List[int]): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation): The output activation function.
        weight_initialization_mode (InitFunction): The initialization function.
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


def set_optimizer(
    opt: str,
    module: nn.Module | list[nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Returns an initialized optimizer from PyTorch.

    .. note::

        The optimizer can be chosen from the following list:

        - Adam
        - AdamW
        - Adadelta
        - Adagrad
        - Adamax
        - ASGD
        - LBFGS
        - RMSprop
        - Rprop
        - SGD

    Args:
        opt (str): optimizer name.
        module (Union[nn.Module, List[nn.Parameter]]): module or parameters.
        learning_rate (float): learning rate.
    """
    assert hasattr(torch.optim, opt), f'Optimizer={opt} not found in torch.'
    optimizer = getattr(torch.optim, opt)

    if isinstance(module, list):
        return optimizer(module, lr=learning_rate)
    if isinstance(module, nn.Module):
        return optimizer(module.parameters(), lr=learning_rate)
    raise TypeError(f'Invalid module type: {type(module)}')
