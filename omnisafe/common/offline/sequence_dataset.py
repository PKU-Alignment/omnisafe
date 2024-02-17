# Copyright 2022-2024 OmniSafe Team. All Rights Reserved.
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
"""Implementation of SequenceDataset."""

import os
import pickle
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class RewardBatch(NamedTuple):
    """A batch of trajectory."""

    trajectories: torch.Tensor
    conditions: Dict[int, torch.Tensor]
    returns: torch.Tensor


class SequenceDataset(Dataset):
    """A dataset class for sequence data used in reinforcement learning."""

    def __init__(
        self,
        dataset_name: str,
        horizon: int = 30,
        max_n_episodes: int = 10000,
        max_path_length: int = 300,
        reward_discount: float = 0.99,
        returns_scale: int = 300,
        device: Optional[torch.device] = None,
    ) -> None:
        """A dataset class for sequence data used in reinforcement learning.

        Args:
            dataset_name (str): The name of the dataset file.
            horizon (int, optional): The length of each sequence. Defaults to 30.
            max_n_episodes (int, optional): The maximum number of episodes to include in the dataset. Defaults to 10000.
            max_path_length (int, optional): The maximum length of each episode. Defaults to 300.
            reward_discount (float, optional): The discount factor for rewards. Defaults to 0.99.
            returns_scale (int, optional): The scaling factor for returns. Defaults to 300.
            device (torch.device, optional): The device to use for tensor operations. Defaults to "cpu".
        """
        device = device or torch.device('cpu')
        self._horizon = horizon
        self._dict = {
            'path_lengths': np.zeros((max_n_episodes,), dtype=np.int64),
        }
        self._count = 0
        self._device = device

        self._max_n_episodes = max_n_episodes
        self._max_path_length = max_path_length
        self._returns_scale = returns_scale

        self._discount = reward_discount
        self._discounts = self._discount ** np.arange(self._max_path_length)[:, None]
        self.keys: List[str] = []

        if os.path.exists(dataset_name) and dataset_name.endswith('.pkl'):
            # Load data from local .npz file
            try:
                with open(dataset_name, 'rb') as f:
                    data = pickle.load(f)  # noqa: S301
            except Exception as e:  # noqa: BLE001
                raise ValueError(f'Failed to load data from {dataset_name}') from e
        else:
            raise ValueError(
                f'Failed to load data from {dataset_name}:'
                'cannot find the file or the extension name is not .pkl',
            )

        for d in data:
            self._add_path(d)

        self._finalize()
        self._indices = self._make_indices(self._dict['path_lengths'], horizon)

    def _add_keys(self, path: dict) -> None:
        """Add keys to the dataset."""
        if self.keys:
            return
        for key in path:
            if key not in ['infos', 'env_infos']:
                self.keys.append(key)

    def _add_path(self, path: dict) -> None:
        """Add a path to the dataset."""
        path_length = len(path['observations'])
        assert path_length <= self._max_path_length
        if self._count >= self._max_n_episodes:
            return

        self._add_keys(path)
        for key in self.keys:
            array = path[key]
            # print(array)
            if key not in self._dict:
                self._dict[key] = np.zeros(
                    (self._max_n_episodes, self._max_path_length, len(array[0])),
                    dtype=np.float32,
                )
            self._dict[key][self._count, :path_length] = array

        self._dict['path_lengths'][self._count] = path_length

        self._count += 1

    def _finalize(self) -> None:
        """Finalize the dataset."""
        # remove extra slots
        for key in [*self.keys, 'path_lengths']:
            self._dict[key] = self._dict[key][: self._count]

    def _make_indices(self, path_lengths: np.ndarray, horizon: int) -> np.ndarray:
        """Makes indices for sampling from dataset; each index maps to a datapoint."""
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - horizon, self._max_path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        return np.array(indices)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._indices)

    def __getitem__(self, idx: Union[int, List[int]]) -> RewardBatch:
        """Get a batch of data from the dataset."""
        if isinstance(idx, int):
            idx = [idx]
        indices = self._indices[idx]
        state_conditions = []
        trajectories = []
        cls_free_conds = []
        for ix in indices:
            path_ind, start, end = ix
            observation = self._dict['observations'][path_ind, start:end]
            action = self._dict['actions'][path_ind, start:end]
            trajectorie = np.concatenate([action, observation], axis=-1)
            trajectories.append(trajectorie)
            state_conditions.append(observation[0])
            cls_free_conds.append(self._dict['cls_free_cond'][path_ind, start])

        trajectories = torch.tensor(np.array(trajectories), device=self._device)
        state_conditions = torch.tensor(np.array(state_conditions), device=self._device)
        cls_free_conds = torch.tensor(np.array(cls_free_conds), device=self._device)

        state_conditions = {0: state_conditions}
        return RewardBatch(trajectories, state_conditions, cls_free_conds)
