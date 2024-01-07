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
"""Test utils."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn
from torch.distributions import Normal

import helpers
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.distributed import fork
from omnisafe.utils.exp_grid_tools import train
from omnisafe.utils.math import (
    SafeTanhTransformer,
    TanhNormal,
    discount_cumsum,
    get_diagonal,
    get_transpose,
)
from omnisafe.utils.model import get_activation, initialize_layer
from omnisafe.utils.schedule import ConstantSchedule, PiecewiseSchedule
from omnisafe.utils.tools import (
    assert_with_exit,
    custom_cfgs_to_dict,
    load_yaml,
    recursive_check_config,
    update_dict,
)


def test_update_dict():
    d = {'a': 1, 'b': {'c': 2}}
    update_dict(d, {'a': 2, 'b': {'d': 3}, 'e': {'f': 4}})
    assert d == {'a': 2, 'b': {'c': 2, 'd': 3}, 'e': {'f': 4}}


def test_assert_with_exit():
    with pytest.raises(SystemExit):
        assert_with_exit(False, 'test')


def test_custom_cfgs_to_dict():
    unparsed_args = {
        'str': 'PPOLag',
        'str_true': 'True',
        'str_false': 'False',
        'float': '1.0',
        'digit': '2',
        'list': '[a,b,c]',
    }
    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    print(custom_cfgs)
    assert custom_cfgs['str'] == unparsed_args['str']
    assert custom_cfgs['str_true'] is True
    assert custom_cfgs['str_false'] is False
    assert custom_cfgs['float'] == float(unparsed_args['float'])
    assert custom_cfgs['digit'] == int(unparsed_args['digit'])
    assert custom_cfgs['list'] == ['a', 'b', 'c']


def test_config():
    """Test config"""
    cfg = Config(a=1, b={'c': 2}, model_cfgs={'actor_type': 'gaussian_learning'})
    cfg.a = 2
    cfg.recurisve_update({'a': {'d': 3}, 'e': {'f': 4}})
    cfg = get_default_kwargs_yaml('PPO', 'Simple-v0', 'on-policy')
    cfg.recurisve_update({'exp_name': 'test_configs', 'env_id': 'Simple-v0', 'algo': 'PPO'})
    check_all_configs(cfg, 'on-policy', 'box')
    with pytest.raises(AssertionError):
        check_all_configs(cfg, 'off-pocliy', 'discrete')
    cfg.recurisve_update({'model_cfgs': {'actor_type': 'discrete'}})
    with pytest.raises(AssertionError):
        check_all_configs(cfg, 'on-pocliy', 'box')


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


def test_train(
    exp_name='make_test_exp_grid',
    algo='CPO',
    env_id='SafetyHalfCheetahVelocity-v1',
):
    """Test train."""
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
            'log_dir': 'saved_log',
        },
    }
    train(
        exp_id=exp_name,
        algo=algo,
        env_id=env_id,
        custom_cfgs=custom_cfgs,
    )
    # delete the saved data
    os.system('rm -rf saved_log')


@helpers.parametrize(
    init_function=['kaiming_uniform', 'xavier_normal', 'glorot', 'xavier_uniform', 'orthogonal'],
)
def test_initalize(init_function: InitFunction):
    """Test initialize."""
    layer = nn.Linear(3, 3)
    initialize_layer(init_function=init_function, layer=layer)

    with pytest.raises(TypeError) as e:
        initialize_layer(init_function='test', layer=layer)
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


def teardown_module():
    """teardown_module."""
    # remove runs folder
    current_path = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(current_path, 'runs')
    if os.path.exists(runs_path):
        os.system(f'rm -rf {runs_path}')

    # remove exp-x folder
    exp_x_path = os.path.join(current_path, 'exp-x')
    if os.path.exists(exp_x_path):
        os.system(f'rm -rf {exp_x_path}')

    # remove png
    os.system(f'rm -rf {current_path}/algo--*.png')


def test_load_yaml():
    not_a_path = 'not_a_path'
    with pytest.raises(FileNotFoundError):
        load_yaml(not_a_path)


def test_recursive_check_config():
    config = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'not_exist': 1}
    default_config = {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': 4}}}
    with pytest.raises(KeyError):
        recursive_check_config(config, default_config)
