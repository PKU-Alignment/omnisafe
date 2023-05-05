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
"""Offline dataset for offline algorithms."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import gdown
import numpy as np
import torch
from torch.utils.data import Dataset

from omnisafe.typing import DEVICE_CPU


@dataclass
class OfflineMeta:
    """Meta information of the offline dataset."""

    episode_length: int | None
    url: str
    sha256sum: str


class OfflineDataset(Dataset):
    """A dataset for offline algorithms."""

    _name_to_metadata: dict[str, OfflineMeta] = {
        'SafetyPointCircle1-v0_mixed_0.5': OfflineMeta(
            url='https://drive.google.com/file/d/1CNHoC70kVIE0wP4VoYy0EH4DmdExGCqM/view?usp=share_link',
            sha256sum='c33e9b102524b26a7466fd542a3e9e925bc5a7eb8a9fdc4a0dc15443819748fd',
            episode_length=500,
        ),
    }
    _default_download_dir = '~/.cache/omnisafe/datasets/'

    def __init__(  # pylint: disable=too-many-branches
        self,
        dataset_name: str,
        batch_size: int = 256,
        gpu_threshold: int = 1024,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_name: The name of the dataset. could be one of the following:
                - 'SafetyPointCircle1-v0_mixed_0.5'
                - some local .npz file
            batch_size: The batch size of the dataset.
            gpu_threshold: The threshold of size(MB) of the dataset to be loaded on GPU.
            device: The device to load the dataset.
        """
        if os.path.exists(dataset_name) and dataset_name.endswith('.npz'):
            # Load data from local .npz file
            try:
                data = np.load(dataset_name)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f'Failed to load data from {dataset_name}') from e

        else:
            # Download .npz file from Google Drive
            url = self._name_to_metadata[dataset_name].url
            sha256sum = self._name_to_metadata[dataset_name].sha256sum

            if not os.path.exists(self._default_download_dir):
                os.makedirs(self._default_download_dir)

            file_path = os.path.join(self._default_download_dir, f'{dataset_name}.npz')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()

                if sha256 == sha256sum:
                    print(f'Dataset {dataset_name} already exists and is valid.')
                else:
                    print(
                        f'Dataset {dataset_name} already exists but is invalid. Downloading again...',
                    )
                    gdown.download(url, file_path, quiet=False, fuzzy=True)

            else:
                print(f'Dataset {dataset_name} does not exist. Downloading...')
                gdown.download(url, file_path, quiet=False, fuzzy=True)

            # Load data from downloaded .npz file
            data = np.load(file_path)

        # Validate the loaded data and convert to tensors
        required_fields = {'obs', 'action', 'reward', 'cost', 'next_obs', 'done'}
        if not all(field in data for field in required_fields):
            raise ValueError(
                f'Loaded data does not have all the required fields: {required_fields}',
            )

        total_size_bytes = 0.0
        for field in required_fields:
            field_size_bytes = data[field].nbytes
            total_size_bytes += field_size_bytes
            print(f"Size of field '{field}': {field_size_bytes / 1024 / 1024:.2f} MB")

        total_size_bytes /= 1024.0 * 1024.0
        print(f'Total size of loaded data: {total_size_bytes:.2f} MB')

        self._batch_size = batch_size
        self._gpu_threshold = gpu_threshold
        self._pre_transfer = False

        # Determine whether to use GPU or not
        if total_size_bytes <= gpu_threshold:
            self._pre_transfer = True

        if self._pre_transfer:
            self.obs = torch.from_numpy(data['obs']).to(device=device)
            self.action = torch.from_numpy(data['action']).to(device=device)
            self.reward = torch.from_numpy(data['reward']).to(device=device)
            self.cost = torch.from_numpy(data['cost']).to(device=device)
            self.next_obs = torch.from_numpy(data['next_obs']).to(device=device)
            self.done = torch.from_numpy(data['done']).to(device=device)
        else:
            self.obs = torch.Tensor(data['obs'])
            self.action = torch.Tensor(data['action'])
            self.reward = torch.Tensor(data['reward'])
            self.cost = torch.Tensor(data['cost'])
            self.next_obs = torch.Tensor(data['next_obs'])
            self.done = torch.Tensor(data['done'])

        self._device = device
        self._length = len(self.obs)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._length

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, ...]:
        """Get a single sample from the dataset.

        Args:
            idx: The index of the sample.

        Returns:
            A tuple of tensors containing the sample.
        """
        if self._pre_transfer:
            return (
                self.obs[idx],
                self.action[idx],
                self.reward[idx],
                self.cost[idx],
                self.next_obs[idx],
                self.done[idx],
            )

        return (
            self.obs[idx].to(device=self._device),
            self.action[idx].to(device=self._device),
            self.reward[idx].to(device=self._device),
            self.cost[idx].to(device=self._device),
            self.next_obs[idx].to(device=self._device),
            self.done[idx].to(device=self._device),
        )

    def sample(
        self,
    ) -> tuple[torch.Tensor, ...]:
        """Sample a batch of data from the dataset."""
        indices = torch.randint(low=0, high=len(self), size=(self._batch_size,), dtype=torch.int64)
        batch_obs = self.obs[indices]
        batch_action = self.action[indices]
        batch_reward = self.reward[indices]
        batch_cost = self.cost[indices]
        batch_next_obs = self.next_obs[indices]
        batch_done = self.done[indices]

        if self._pre_transfer:
            return (batch_obs, batch_action, batch_reward, batch_cost, batch_next_obs, batch_done)

        return (
            batch_obs.to(device=self._device),
            batch_action.to(device=self._device),
            batch_reward.to(device=self._device),
            batch_cost.to(device=self._device),
            batch_next_obs.to(device=self._device),
            batch_done.to(device=self._device),
        )


class OfflineDatasetWithInit(OfflineDataset):
    """A dataset with first observation in every episodes for offline algorithms."""

    def __init__(  # pylint: disable=too-many-branches, super-init-not-called
        self,
        dataset_name: str,
        batch_size: int = 256,
        gpu_threshold: int = 1024,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_name: The name of the dataset. could be one of the following:
                - 'SafetyPointCircle1-v0_mixed_0.5'
                - some local .npz file
            batch_size: The batch size of the dataset.
            gpu_threshold: The threshold of size(MB) of the dataset to be loaded on GPU.
            device: The device to load the dataset.
        """
        if os.path.exists(dataset_name) and dataset_name.endswith('.npz'):
            # Load data from local .npz file
            try:
                data = np.load(dataset_name)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f'Failed to load data from {dataset_name}') from e

        else:
            # Download .npz file from Google Drive
            url = self._name_to_metadata[dataset_name].url
            sha256sum = self._name_to_metadata[dataset_name].sha256sum

            if not os.path.exists(self._default_download_dir):
                os.makedirs(self._default_download_dir)

            file_path = os.path.join(self._default_download_dir, f'{dataset_name}.npz')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()

                if sha256 == sha256sum:
                    print(f'Dataset {dataset_name} already exists and is valid.')
                else:
                    print(
                        f'Dataset {dataset_name} already exists but is invalid. Downloading again...',
                    )
                    gdown.download(url, file_path, quiet=False, fuzzy=True)

            else:
                print(f'Dataset {dataset_name} does not exist. Downloading...')
                gdown.download(url, file_path, quiet=False, fuzzy=True)

            # Load data from downloaded .npz file
            data = np.load(file_path)

        # Validate the loaded data and convert to tensors
        required_fields = {'obs', 'action', 'reward', 'cost', 'next_obs', 'done'}
        if not all(field in data for field in required_fields):
            raise ValueError(
                f'Loaded data does not have all the required fields: {required_fields}',
            )

        try:
            episode_length = self._name_to_metadata[dataset_name].episode_length
        except KeyError:
            episode_length = None
        if episode_length is None:
            try:
                init_obs = data['init_obs']
            except KeyError as e:
                raise ValueError(
                    'Loaded data does not have the required field "init_obs" for episodic data.',
                ) from e
        else:
            init_obs = data['obs'][::episode_length]
            init_obs = np.repeat(init_obs, episode_length, axis=0)

        total_size_bytes = 0.0
        for field in required_fields:
            field_size_bytes = data[field].nbytes
            total_size_bytes += field_size_bytes
            print(f"Size of field '{field}': {field_size_bytes / 1024 / 1024:.2f} MB")
        field_size_bytes = init_obs.nbytes
        total_size_bytes += field_size_bytes
        print(f"Size of field 'init_obs': {field_size_bytes / 1024 / 1024:.2f} MB")

        total_size_bytes /= 1024.0 * 1024.0
        print(f'Total size of loaded data: {total_size_bytes:.2f} MB')

        self._batch_size = batch_size
        self._gpu_threshold = gpu_threshold
        self._pre_transfer = False

        # Determine whether to use GPU or not
        if total_size_bytes <= gpu_threshold:
            self._pre_transfer = True

        if self._pre_transfer:
            self.obs = torch.from_numpy(data['obs']).to(device=device)
            self.action = torch.from_numpy(data['action']).to(device=device)
            self.reward = torch.from_numpy(data['reward']).to(device=device)
            self.cost = torch.from_numpy(data['cost']).to(device=device)
            self.next_obs = torch.from_numpy(data['next_obs']).to(device=device)
            self.done = torch.from_numpy(data['done']).to(device=device)
            self.init_obs = torch.from_numpy(init_obs).to(device=device)
        else:
            self.obs = torch.Tensor(data['obs'])
            self.action = torch.Tensor(data['action'])
            self.reward = torch.Tensor(data['reward'])
            self.cost = torch.Tensor(data['cost'])
            self.next_obs = torch.Tensor(data['next_obs'])
            self.done = torch.Tensor(data['done'])
            self.init_obs = torch.Tensor(init_obs)

        self._device = device
        self._length = len(self.obs)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._length

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, ...]:
        """Get a single sample from the dataset.

        Args:
            idx: The index of the sample.

        Returns:
            A tuple of tensors containing the sample.
        """
        if self._pre_transfer:
            return (
                self.obs[idx],
                self.action[idx],
                self.reward[idx],
                self.cost[idx],
                self.next_obs[idx],
                self.done[idx],
                self.init_obs[idx],
            )

        return (
            self.obs[idx].to(device=self._device),
            self.action[idx].to(device=self._device),
            self.reward[idx].to(device=self._device),
            self.cost[idx].to(device=self._device),
            self.next_obs[idx].to(device=self._device),
            self.done[idx].to(device=self._device),
            self.init_obs[idx].to(device=self._device),
        )

    def sample(
        self,
    ) -> tuple[torch.Tensor, ...]:
        """Sample a batch of data from the dataset."""
        indices = torch.randint(low=0, high=len(self), size=(self._batch_size,), dtype=torch.int64)
        batch_obs = self.obs[indices]
        batch_action = self.action[indices]
        batch_reward = self.reward[indices]
        batch_cost = self.cost[indices]
        batch_next_obs = self.next_obs[indices]
        batch_done = self.done[indices]
        barch_init_obs = self.init_obs[indices]

        if self._pre_transfer:
            return (
                batch_obs,
                batch_action,
                batch_reward,
                batch_cost,
                batch_next_obs,
                batch_done,
                barch_init_obs,
            )

        return (
            batch_obs.to(device=self._device),
            batch_action.to(device=self._device),
            batch_reward.to(device=self._device),
            batch_cost.to(device=self._device),
            batch_next_obs.to(device=self._device),
            batch_done.to(device=self._device),
            barch_init_obs.to(device=self._device),
        )
