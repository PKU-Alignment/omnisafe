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
"""Test utils"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from torch import nn
from torch.distributions import Normal

import helpers
import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.distributed import fork
from omnisafe.utils.math import (
    SafeTanhTransformer,
    TanhNormal,
    discount_cumsum,
    get_diagonal,
    get_transpose,
)
from omnisafe.utils.model import get_activation, initialize_layer
from omnisafe.utils.schedule import ConstantSchedule, PiecewiseSchedule


def test_config():
    """Test config"""
    cfg = Config(a=1, b={'c': 2})
    cfg.a = 2
    cfg.recurisve_update({'a': {'d': 3}, 'e': {'f': 4}})

    cfg = get_default_kwargs_yaml('PPO', 'Simple-v0', 'on-policy')
    check_all_configs(cfg, 'on-policy')


def test_distributed():
    fork(parallel=2, manual_args=['distribution_train.py'])


def get_answer(gamma: float):
    """Input gamma and return the answer."""
    if gamma == 0.9:
        return torch.tensor([11.4265, 11.5850, 10.6500, 8.5000, 5.0000], dtype=torch.float64)
    if gamma == 0.99:
        return torch.tensor([14.6045, 13.7419, 11.8605, 8.9500, 5.0000], dtype=torch.float64)
    if gamma == 0.999:
        return torch.tensor([14.9600, 13.9740, 11.9860, 8.9950, 5.0000], dtype=torch.float64)
    return None


@helpers.parametrize(
    discount=[0.9, 0.99, 0.999],
)
def test_discount_cumsum_torch(
    discount: float,
):
    """Test discount_cumsum_torch."""
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y1 = get_answer(discount)
    assert torch.allclose(discount_cumsum(x1, discount), y1), 'discount_cumsum_torch is not correct'


def test_TanhNormal():
    """Test TanhNormal."""
    normal = Normal(0, 1)
    tanh_normal = TanhNormal(0, 1)
    assert torch.tanh(normal.mean) == tanh_normal.mean
    assert normal.stddev == tanh_normal.stddev
    assert normal.entropy() == tanh_normal.entropy()
    assert normal.variance == tanh_normal.variance
    tanh_normal = tanh_normal.expand((2, 3))
    assert tanh_normal.sample().shape == (2, 3)


def test_math():
    """Test math."""
    random_tensor = torch.randn(3, 3)
    assert torch.allclose(get_diagonal(random_tensor), torch.diag(random_tensor).sum(-1))
    assert torch.allclose(get_transpose(random_tensor), random_tensor.t())

    random_tensor = torch.clamp(random_tensor, -0.999999, 0.999999)
    tanh = SafeTanhTransformer()
    assert torch.allclose(tanh(random_tensor), torch.tanh(random_tensor))
    assert torch.allclose(tanh.inv(random_tensor), torch.atanh(random_tensor))


def train(
    exp_id: str,
    algo: str,
    env_id: str,
    custom_cfgs: Config,
    num_threads: int = 6,
) -> tuple[float, float, float]:
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
            os.makedirs(custom_cfgs['data_dir'], exist_ok=True)
        sys.stdout = open(  # noqa: SIM115
            f'{custom_cfgs["data_dir"]}terminal.log',
            'w',
            encoding='utf-8',
        )
        sys.stderr = open(  # noqa: SIM115
            f'{custom_cfgs["data_dir"]}error.log',
            'w',
            encoding='utf-8',
        )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


def test_train(
    exp_name='make_test_exp_grid',
    algo='CPO',
    env_id='SafetyHalfCheetahVelocity-v1',
):
    """Test train."""
    eg = ExperimentGrid(exp_name=exp_name)
    eg.add('algo', [algo])
    eg.add('env_id', [env_id])
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('algo_cfgs:update_cycle', [512])
    eg.add('train_cfgs:total_steps', [1024, 2048])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.run(train, num_pool=1, is_test=True)


@helpers.parametrize(
    init_function=['kaiming_uniform', 'xavier_normal', 'glorot', 'xavier_uniform', 'orthogonal'],
)
def test_initalize(init_function: InitFunction):
    """Test initialize."""
    layer = nn.Linear(3, 3)
    initialize_layer(init_function=init_function, layer=layer)

    with pytest.raises(TypeError) as e:
        initialize_layer(init_function='test', layer=layer)  # type: ignore
    assert e.value.args[0] == 'Invalid initialization function: test'


@helpers.parametrize(activations=['identity', 'relu', 'sigmoid', 'softplus', 'tanh'])
def test_get_activation(activations: Activation):
    """Test get_activation."""
    get_activation(activations)


def test_schedule():
    """Test schedule."""
    constant = ConstantSchedule(1)
    assert constant.value(1) == 1

    endpoints = [(0, 0.0), (2, 1.0)]
    piece = PiecewiseSchedule(endpoints, outside_value=0)
    assert piece.value(1) == 0.5
    assert piece.value(100) == 0
