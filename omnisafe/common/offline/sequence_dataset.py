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


# ruff: noqa

import os
import pickle
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset


RewardBatch = namedtuple('Batch', 'trajectories conditions returns')

import numpy as np


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        horizon=30,
        max_n_episodes=10000,
        max_path_length=300,
        reward_discount=0.99,
        returns_scale=300,
        device: torch.device = "cpu",
    ) -> None:
        self._horizon = horizon
        self._dict = {
            'path_lengths': torch.zeros(max_n_episodes, dtype=torch.int64),
        }
        self._count = 0
        self._device = device

        self._max_n_episodes = max_n_episodes
        self._max_path_length = max_path_length
        self._returns_scale = returns_scale

        self._discount = reward_discount
        self._discounts = self._discount ** np.arange(self._max_path_length)[:, None]

        if os.path.exists(dataset_name) and dataset_name.endswith('.pkl'):
            # Load data from local .npz file
            try:
                with open(dataset_name, 'rb') as dataset_name:
                    data = pickle.load(dataset_name)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f'Failed to load data from {dataset_name}') from e
        else:
            raise ValueError(
                f'Failed to load data from {dataset_name}:'
                'cannot find the file or the extension name is not .pkl'
            )

        for i in range(len(data)):
            self._add_path(data[i])

        self._finalize()
        self._indices = self._make_indices(self._dict["path_lengths"], horizon)

    def _add_keys(self, path):
        self.keys = []
        if self.keys:
            return
        for key in path.keys():
            if key not in ['infos', 'env_infos']:
                self.keys.append(key)

    def _add_path(self, path):
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

    def _finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][: self._count]

    def _make_indices(self, path_lengths, horizon):
        '''
        makes indices for sampling from dataset;
        each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - horizon, self._max_path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        indexs = self._indices[idx]
        state_conditions = []
        trajectories = []
        cls_free_conds = []
        for i in range(len(indexs)):
            path_ind, start, end = indexs[i]
            observation = self._dict["observations"][path_ind, start:end]
            action = self._dict["actions"][path_ind, start:end]
            trajectorie = np.concatenate([action, observation], axis=-1)
            trajectories.append(trajectorie)
            state_conditions.append(observation[0])
            cls_free_conds.append(self._dict["cls_free_cond"][path_ind, start])

        trajectories = torch.tensor(np.array(trajectories), device=self._device)
        state_conditions = torch.tensor(np.array(state_conditions), device=self._device)
        cls_free_conds = torch.tensor(np.array(cls_free_conds), device=self._device)

        state_conditions = {0: state_conditions}
        batch = RewardBatch(trajectories, state_conditions, cls_free_conds)
        return batch
