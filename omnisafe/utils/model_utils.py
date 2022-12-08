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
"""This module contains the helper functions for the model."""


from typing import List, Literal

import numpy as np
from torch import nn


# typing
Activation = Literal['identity', 'relu', 'sigmoid', 'softplus', 'tanh']
InitFunction = Literal['kaiming_uniform', 'xavier_normal', 'glorot', 'xavier_uniform', 'orthogonal']


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

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


def get_activation(activation: Activation) -> nn.Module:
    """Get the activation function.

    Args:
        activation (Activation): The activation function.

    Returns:
        nn.Module: The activation function.
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
    sizes: List[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    """Build the MLP network.

    Args:
        sizes (List[int]): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function.
            Defaults to 'identity'.
        weight_initialization_mode (InitFunction, optional): The initialization
            function. Defaults to 'kaiming_uniform'.

    Returns:
        nn.Module: The MLP network.
    """
    activation = get_activation(activation)
    output_activation = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)
