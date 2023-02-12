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
"""Test Utils"""

import os
import sys

import numpy as np
import torch

import helpers
import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple
from omnisafe.utils.core import discount_cumsum_torch
from omnisafe.utils.distributed import dist_statistics_scalar, fork
from omnisafe.utils.tools import to_ndarray


@helpers.parametrize(item=[1, 1.0, [1, 2, 3], (1, 2, 3), {'a': 1, 'b': 2}, torch.tensor([1, 2, 3])])
def test_to_ndarray(item):
    """Test to_ndarray."""
    if isinstance(item, torch.Tensor):
        assert isinstance(to_ndarray(item), np.ndarray)
    elif isinstance(item, list):
        out_list = to_ndarray(item)
        for val in out_list:
            assert isinstance(val, np.ndarray)
    elif isinstance(item, tuple):
        out_tuple = to_ndarray(item)
        for val in out_tuple:
            assert isinstance(val, np.ndarray)
    elif isinstance(item, dict):
        out_dict = to_ndarray(item)
        for val in out_dict.values():
            assert isinstance(val, np.ndarray)
    else:
        assert isinstance(to_ndarray(item), np.ndarray)


def get_answer(gamma: float) -> torch.Tensor:
    """Input gamma and return the answer."""
    if gamma == 0.9:
        return torch.tensor([11.4265, 11.5850, 10.6500, 8.5000, 5.0000], dtype=torch.float64)
    elif gamma == 0.99:
        return torch.tensor([14.6045, 13.7419, 11.8605, 8.9500, 5.0000], dtype=torch.float64)
    elif gamma == 0.999:
        return torch.tensor([14.9600, 13.9740, 11.9860, 8.9950, 5.0000], dtype=torch.float64)


@helpers.parametrize(
    discount=[0.9, 0.99, 0.999],
)
def test_discount_cumsum_torch(
    discount: float,
):
    """Test discount_cumsum_torch."""
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y1 = get_answer(discount)
    assert torch.allclose(
        discount_cumsum_torch(x1, discount), y1
    ), 'discount_cumsum_torch is not correct'


def test_distributed_tools():
    """Test mpi_fork."""
    fork(2, test_message=['examples/train_from_custom_dict.py', '--parallel', '2'])


def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple, num_threads: int = 6
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    torch.set_num_threads(num_threads)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    USE_REDIRECTION = True
    if USE_REDIRECTION:
        if not os.path.exists(custom_cfgs['data_dir']):
            os.makedirs(custom_cfgs['data_dir'])
        sys.stdout = open(f'{custom_cfgs["data_dir"]}terminal.log', 'w', encoding='utf-8')
        sys.stderr = open(f'{custom_cfgs["data_dir"]}error.log', 'w', encoding='utf-8')
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


def test_train(
    exp_name='Safety_Gymnasium_Goal',
    algo='CPO',
    env_id='SafetyHalfCheetahVelocity-v4',
    epochs=1,
    steps_per_epoch=1000,
    num_envs=1,
):
    """Test train."""
    eg = ExperimentGrid(exp_name=exp_name)
    eg.add('algo', [algo])
    eg.add('env_id', [env_id])
    eg.add('epochs', [epochs])
    eg.add('steps_per_epoch', [steps_per_epoch])
    eg.add('env_cfgs', [{'num_envs': num_envs}])
    eg.run(train, num_pool=1, is_test=True)
