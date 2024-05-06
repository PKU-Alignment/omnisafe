# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Utils for CRABS."""
# pylint: disable=all
from __future__ import annotations

import os

import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
from torch import load

from omnisafe.common.control_barrier_function.crabs.models import (
    EnsembleModel,
    GatedTransitionModel,
    TransitionModel,
)


class Normalizer(nn.Module):
    """Normalizes input data to have zero mean and unit variance.

    Args:
        dim (int): Dimension of the input data.
        clip (float): Clip the standard deviation to this value.
    """

    def __init__(self, dim, *, clip=10) -> None:
        """Initialize the normalizer."""
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('n', torch.tensor(0, dtype=torch.int64))
        self.placeholder = nn.Parameter(torch.tensor(0.0), False)  # for device info (@maybe_numpy)
        self.clip = clip

    def forward(self, x, inverse=False):
        """Normalize input data.

        Args:
            x (torch.Tensor): Input data.
            inverse (bool): If True, unnormalize the data.

        Returns:
            torch.Tensor: Normalized data.
        """
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / self.std.clamp(min=1e-6)

    def update(self, data):
        """Update the normalizer with new data.

        Args:
            data (torch.Tensor): New data.
        """
        data = data - self.mean

        m = data.shape[0]
        delta = data.mean(dim=0)
        new_n = self.n + m
        new_mean = self.mean + delta * m / new_n
        new_std = torch.sqrt(
            (self.std**2 * self.n + data.var(dim=0) * m + delta**2 * self.n * m / new_n) / new_n,
        )

        self.mean.set_(new_mean.data)
        self.std.set_(new_std.data)
        self.n.set_(new_n.data)

    def fit(self, data):
        """Fit the normalizer with new data.

        Args:
            data (torch.Tensor): New data.
        """
        n = data.shape[0]
        self.n.set_(torch.tensor(n, device=self.n.device))
        self.mean.set_(data.mean(dim=0))
        self.std.set_(data.std(dim=0))


def download_model(url, destination):
    """Download model from the cloud.

    Args:
        url (str): URL of the model.
        destination (str): Path to save the model.
    """
    response = requests.get(url)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        f.write(response.content)


def get_pretrained_model(model_path, model_url, device):
    """Get pretrained model.

    If the model is not found locally, download it from the cloud.

    Args:
        model_path (str): Path to save the model.
        model_url (str): URL of the model.
        device (torch.device): Device to load the model.

    Returns:
        torch.nn.Module: Pretrained model.
    """
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print('Model not found locally. Downloading from cloud...')
        download_model(model_url, model_path)
    else:
        print('Model found locally.')

    return load(model_path, map_location=device)


def create_model_and_trainer(cfgs, dim_state, dim_action, normalizer, device):
    """Create world model and trainer.

    Args:
        cfgs: Configs.
        dim_state: Dimension of the state.
        dim_action: Dimension of the action.
        normalizer: Observation normalizer.
        device: Device to load the model.

    Returns:
        Tuple[nn.Module, pl.Trainer]: World model and trainer.
    """

    def make_model(i, model_type) -> nn.Module:
        if model_type == 'GatedTransitionModel':
            return GatedTransitionModel(
                dim_state,
                normalizer,
                [dim_state + dim_action, 256, 256, 256, 256, dim_state * 2],
                cfgs.transition_model_cfgs.train,
                name=f'model-{i}',
            )
        if model_type == 'TransitionModel':
            return TransitionModel(
                dim_state,
                normalizer,
                [dim_state + dim_action, 256, 256, 256, 256, dim_state * 2],
                cfgs.transition_model_cfgs.train,
                name=f'model-{i}',
            )
        raise AssertionError(f'unknown model type {model_type}')

    model_type = cfgs.transition_model_cfgs.type
    models = [make_model(i, model_type) for i in range(cfgs.transition_model_cfgs.n_ensemble)]

    model = EnsembleModel(models).to(device)

    devices: list[int] | int

    if str(device).startswith('cuda'):
        accelerator = 'gpu'
        devices = [int(str(device)[-1])]
    else:
        accelerator = 'cpu'
        devices = torch.get_num_threads()
    trainer = pl.Trainer(
        max_epochs=0,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=cfgs.logger_cfgs.log_dir,
    )

    return model, trainer
