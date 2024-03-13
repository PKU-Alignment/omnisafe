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
"""Test offline module."""
from __future__ import annotations

import os

import torch

from omnisafe.common.offline.data_collector import OfflineDataCollector
from omnisafe.common.offline.dataset import OfflineDataset, OfflineDatasetWithInit


def test_data_collector():
    env_name = 'SafetyPointGoal1-v0'
    size = 2_000
    base_dir = os.path.dirname(__file__)
    agents = [
        (
            os.path.join(
                base_dir,
                'saved_source',
                'PPO-{SafetyPointGoal1-v0}',
                'seed-000-2023-03-16-12-08-52',
            ),
            'epoch-0.pt',
            2_000,
        ),
    ]
    save_dir = os.path.join(base_dir, 'saved_data')

    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)
    # delete the saved data
    os.system(f'rm -rf {save_dir}')


def check_dataset(dataset: OfflineDataset | OfflineDatasetWithInit):
    (obs, action, reward, cost, next_obs, done) = dataset.sample()
    assert isinstance(obs, torch.Tensor) and obs.shape == torch.Size([256, 60])
    assert isinstance(next_obs, torch.Tensor) and next_obs.shape == torch.Size([256, 60])
    assert isinstance(action, torch.Tensor) and action.shape == torch.Size([256, 2])
    assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([256, 1])
    assert isinstance(cost, torch.Tensor) and cost.shape == torch.Size([256, 1])
    assert isinstance(done, torch.Tensor) and done.shape == torch.Size([256, 1])
    (obs, action, reward, cost, next_obs, done) = dataset[0]
    assert isinstance(obs, torch.Tensor) and obs.shape == torch.Size([60])
    assert isinstance(next_obs, torch.Tensor) and next_obs.shape == torch.Size([60])
    assert isinstance(action, torch.Tensor) and action.shape == torch.Size([2])
    assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([1])
    assert isinstance(cost, torch.Tensor) and cost.shape == torch.Size([1])
    assert isinstance(done, torch.Tensor) and done.shape == torch.Size([1])


def check_init_dataset(dataset: OfflineDataset | OfflineDatasetWithInit):
    (obs, action, reward, cost, next_obs, done, init_obs) = dataset.sample()
    assert isinstance(obs, torch.Tensor) and obs.shape == torch.Size([256, 60])
    assert isinstance(next_obs, torch.Tensor) and next_obs.shape == torch.Size([256, 60])
    assert isinstance(action, torch.Tensor) and action.shape == torch.Size([256, 2])
    assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([256, 1])
    assert isinstance(cost, torch.Tensor) and cost.shape == torch.Size([256, 1])
    assert isinstance(done, torch.Tensor) and done.shape == torch.Size([256, 1])
    assert isinstance(init_obs, torch.Tensor) and init_obs.shape == torch.Size([256, 60])
    (obs, action, reward, cost, next_obs, done, init_obs) = dataset[0]
    assert isinstance(obs, torch.Tensor) and obs.shape == torch.Size([60])
    assert isinstance(next_obs, torch.Tensor) and next_obs.shape == torch.Size([60])
    assert isinstance(action, torch.Tensor) and action.shape == torch.Size([2])
    assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([1])
    assert isinstance(cost, torch.Tensor) and cost.shape == torch.Size([1])
    assert isinstance(done, torch.Tensor) and done.shape == torch.Size([1])
    assert isinstance(init_obs, torch.Tensor) and init_obs.shape == torch.Size([60])


def test_url_link():
    dataset = OfflineDataset(
        dataset_name='SafetyPointGoal1-v0_data_test',
        gpu_threshold=0.0,
    )
    dataset_transfered = OfflineDataset(
        dataset_name='SafetyPointGoal1-v0_data_test',
        gpu_threshold=99999.9,
    )
    dataset_init = OfflineDatasetWithInit(
        dataset_name='SafetyPointGoal1-v0_data_init_test',
        gpu_threshold=0.0,
    )
    dataset_init_transfered = OfflineDatasetWithInit(
        dataset_name='SafetyPointGoal1-v0_data_init_test',
        gpu_threshold=99999.9,
    )
    check_dataset(dataset=dataset)
    check_dataset(dataset=dataset_transfered)
    check_init_dataset(dataset=dataset_init)
    check_init_dataset(dataset=dataset_init_transfered)
    # delete the saved data

    os.system('rm -rf ./~')
