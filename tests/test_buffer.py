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
"""Test Buffers."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

import helpers
from omnisafe.common.buffer import (
    OffPolicyBuffer,
    OnPolicyBuffer,
    VectorOffPolicyBuffer,
    VectorOnPolicyBuffer,
)


@helpers.parametrize(
    obs_space=[Box(low=-1, high=1, shape=(1,)), Discrete(n=5)],
    act_space=[Box(low=-1, high=1, shape=(1,)), Discrete(n=5)],
    size=[100],
    gamma=[0.9],
    lam=[0.9],
    advantage_estimator=['gae', 'vtrace', 'gae-rtg', 'plain'],
    standardized_adv_r=[True],
    standardized_adv_c=[True],
    lam_c=[0.9],
    penalty_coefficient=[0.0],
    device=[torch.device('cpu')],
    num_envs=[2],
)
def test_vector_onpolicy_buffer(
    obs_space: Box | Discrete,
    act_space: Box | Discrete,
    size: int,
    gamma: float,
    lam: float,
    advantage_estimator: str,
    standardized_adv_r: bool,
    standardized_adv_c: bool,
    lam_c: float,
    penalty_coefficient: float,
    device: str,
    num_envs: int,
) -> None:
    """Test the VectorBuffer class."""
    vector_buffer = VectorOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=size,
        gamma=gamma,
        lam=lam,
        advantage_estimator=advantage_estimator,
        standardized_adv_r=standardized_adv_r,
        standardized_adv_c=standardized_adv_c,
        lam_c=lam_c,
        penalty_coefficient=penalty_coefficient,
        device=device,
        num_envs=num_envs,
    )
    # checking the initialized value
    assert (
        vector_buffer.num_buffers == num_envs
    ), f'vector_buffer.num_buffers is {vector_buffer.num_buffers}'
    assert (
        vector_buffer.standardized_adv_c == standardized_adv_c
    ), f'vector_buffer.standardized_adv_c is {vector_buffer.standardized_adv_c}'
    assert (
        vector_buffer.standardized_adv_r == standardized_adv_r
    ), f'vector_buffer.sstandardized_adv_r is {vector_buffer.sstandardized_adv_r}'
    assert vector_buffer.buffers is not [], f'vector_buffer.buffers is {vector_buffer.buffers}'

    # checking the store function
    obs_dim = int(np.array(obs_space.shape).prod())
    act_dim = int(np.array(act_space.shape).prod())
    for _ in range(size):
        obs = torch.rand((num_envs, obs_dim), dtype=torch.float32, device=device)
        act = torch.rand((num_envs, act_dim), dtype=torch.float32, device=device)
        reward = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        cost = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        value_r = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        value_c = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        logp = torch.rand((num_envs, 1), dtype=torch.float32, device=device)

        vector_buffer.store(
            obs=obs,
            act=act,
            reward=reward,
            cost=cost,
            value_r=value_r,
            value_c=value_c,
            logp=logp,
        )
        for idx, buffer in enumerate(vector_buffer.buffers):
            assert torch.allclose(
                buffer.data['obs'][buffer.ptr - 1],
                obs[idx],
            ), f'buffer.data[obs][buffer.ptr - 1] is {buffer.data["obs"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['act'][buffer.ptr - 1],
                act[idx],
            ), f'buffer.data[act][buffer.ptr - 1] is {buffer.data["act"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['reward'][buffer.ptr - 1],
                reward[idx],
            ), f'buffer.data[reward][buffer.ptr - 1] is {buffer.data["reward"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['cost'][buffer.ptr - 1],
                cost[idx],
            ), f'buffer.data[cost][buffer.ptr - 1] is {buffer.data["cost"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['value_r'][buffer.ptr - 1],
                value_r[idx],
            ), f'buffer.data[value_r][buffer.ptr - 1] is {buffer.data["value_r"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['value_c'][buffer.ptr - 1],
                value_c[idx],
            ), f'buffer.data[value_c][buffer.ptr - 1] is {buffer.data["value_c"][buffer.ptr - 1]}'
            assert torch.allclose(
                buffer.data['logp'][buffer.ptr - 1],
                logp[idx],
            ), f'buffer.data[logp][buffer.ptr - 1] is {buffer.data["logp"][buffer.ptr - 1]}'

    # checking the finish_path function
    for idx, buffer in enumerate(vector_buffer.buffers):
        last_value_r = torch.randn(1, device=device)
        last_value_c = torch.randn(1, device=device)
        vector_buffer.finish_path(last_value_r, last_value_c, idx)
        assert (
            buffer.path_start_idx == buffer.ptr
        ), f'buffer.path_start_idx is {buffer.path_start_idx}'

    # checking the get function
    data = vector_buffer.get()
    print(data['obs'].shape)
    assert data['obs'].shape == (
        size * num_envs,
        obs_dim,
    ), f'data[obs].shape is {data["obs"].shape}'
    assert data['act'].shape == (
        size * num_envs,
        act_dim,
    ), f'data[act].shape is {data["act"].shape}'


@helpers.parametrize(
    obs_space=[Box(low=-1, high=1, shape=(1,))],
    act_space=[Box(low=-1, high=1, shape=(1,))],
    size=[100],
    gamma=[0.9],
    lam=[0.9],
    advantage_estimator=['gae', 'vtrace', 'gae-rtg', 'plain'],
    standardized_adv_r=[True],
    standardized_adv_c=[True],
    lam_c=[0.9],
    penalty_coefficient=[0.0],
    device=[torch.device('cpu')],
)
def test_onpolicy_buffer(
    obs_space: Box,
    act_space: Box,
    size: int,
    gamma: float,
    lam: float,
    advantage_estimator: str,
    standardized_adv_r: bool,
    standardized_adv_c: bool,
    lam_c: float,
    penalty_coefficient: float,
    device: str,
) -> None:
    """Test buffer."""
    # initialize buffer
    buffer = OnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=size,
        gamma=gamma,
        lam=lam,
        advantage_estimator=advantage_estimator,
        lam_c=lam_c,
        penalty_coefficient=penalty_coefficient,
        standardized_adv_r=standardized_adv_r,
        standardized_adv_c=standardized_adv_c,
        device=device,
    )
    # checking the initialized value
    assert buffer.ptr == 0, f'buffer.ptr is {buffer.ptr}'
    assert buffer.path_start_idx == 0, f'buffer.path_start_idx is {buffer.path_start_idx}'
    assert len(buffer) == size, f'buffer.size is {buffer.size}'
    assert buffer.max_size == size, f'buffer.max_size is {buffer.max_size}'
    assert buffer.device == device, f'buffer.device is {buffer.device}'
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    expected_data = {
        'obs': torch.zeros((size, obs_dim), dtype=torch.float32, device=device),
        'act': torch.zeros((size, act_dim), dtype=torch.float32, device=device),
        'logp': torch.zeros((size), dtype=torch.float32, device=device),
    }
    # check the store function
    for _ in range(size):
        obs, act, reward, value_r, logp, cost, value_c = (
            torch.randn(obs_dim, device=device),
            torch.randn(act_dim, device=device),
            torch.randn(1, device=device),
            torch.randn(1, device=device),
            torch.randn(1, device=device),
            torch.randn(1, device=device),
            torch.randn(1, device=device),
        )
        expected_data['obs'][buffer.ptr] = obs
        expected_data['act'][buffer.ptr] = act
        expected_data['logp'][buffer.ptr] = logp
        buffer.store(
            obs=obs,
            act=act,
            reward=reward,
            value_r=value_r,
            logp=logp,
            cost=cost,
            value_c=value_c,
        )
        assert torch.allclose(
            buffer.data['obs'][buffer.ptr - 1],
            obs,
        ), f'buffer.data[obs][buffer.ptr-1] is {buffer.data["obs"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['act'][buffer.ptr - 1],
            act,
        ), f'buffer.data[act][buffer.ptr-1] is {buffer.data["act"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['reward'][buffer.ptr - 1],
            reward,
        ), f'buffer.data[reward][buffer.ptr-1] is {buffer.data["reward"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['value_r'][buffer.ptr - 1],
            value_r,
        ), f'buffer.data[value_r][buffer.ptr-1] is {buffer.data["value_r"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['logp'][buffer.ptr - 1],
            logp,
        ), f'buffer.data[logp][buffer.ptr-1] is {buffer.data["logp"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['cost'][buffer.ptr - 1],
            cost,
        ), f'buffer.data[cost][buffer.ptr-1] is {buffer.data["cost"][buffer.ptr-1]}'
        assert torch.allclose(
            buffer.data['value_c'][buffer.ptr - 1],
            value_c,
        ), f'buffer.data[value_c][buffer.ptr-1] is {buffer.data["value_c"][buffer.ptr-1]}'

    assert buffer.ptr == size, f'buffer.ptr is {buffer.ptr}'

    # check the finish_path function
    last_value_r = torch.randn(1, device=device)
    last_value_c = torch.randn(1, device=device)
    buffer.finish_path(last_value_r=last_value_r, last_value_c=last_value_c)
    assert buffer.path_start_idx == buffer.ptr, f'buffer.path_start_idx is {buffer.path_start_idx}'

    # check the get function
    data = buffer.get()
    assert torch.allclose(data['obs'], expected_data['obs'])
    assert torch.allclose(data['act'], expected_data['act'])
    assert torch.allclose(data['logp'], expected_data['logp'])
    assert buffer.ptr == 0, f'buffer.ptr is {buffer.ptr}'
    assert buffer.path_start_idx == 0, f'buffer.path_start_idx is {buffer.path_start_idx}'


@helpers.parametrize(
    obs_space=[Box(low=-1, high=1, shape=(1,))],
    act_space=[Box(low=-1, high=1, shape=(1,))],
    size=[10],
    batch_size=[5],
    device=[torch.device('cpu')],
    num_envs=[2],
)
def test_vector_offpolicy_buffer(
    obs_space: Box,
    act_space: Box,
    size: int,
    batch_size: int,
    device: str,
    num_envs: int,
) -> None:
    """Test buffer."""
    # initialize buffer
    buffer = VectorOffPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=size,
        batch_size=batch_size,
        device=device,
        num_envs=num_envs,
    )
    # checking the initialized value
    assert buffer.max_size == size, f'buffer.max_size is {buffer.max_size}'
    assert buffer.batch_size == batch_size, f'buffer.batch_size is {buffer.batch_size}'
    assert buffer.device == device, f'buffer.device is {buffer.device}'
    assert buffer.num_envs == num_envs, f'buffer.num_envs is {buffer.num_envs}'

    expected_data = {
        'obs': torch.randn((size, num_envs, *obs_space.shape), dtype=torch.float32, device=device),
        'act': torch.randn((size, num_envs, *act_space.shape), dtype=torch.float32, device=device),
        'reward': torch.randn((size, num_envs), dtype=torch.float32, device=device),
        'cost': torch.randn((size, num_envs), dtype=torch.float32, device=device),
        'done': torch.randn((size, num_envs), dtype=torch.float32, device=device),
        'next_obs': torch.randn(
            (size, num_envs, *obs_space.shape),
            dtype=torch.float32,
            device=device,
        ),
    }
    # check the store function
    for i in range(size):
        buffer.store(
            obs=expected_data['obs'][i],
            act=expected_data['act'][i],
            reward=expected_data['reward'][i],
            cost=expected_data['cost'][i],
            done=expected_data['done'][i],
            next_obs=expected_data['next_obs'][i],
        )

    assert torch.allclose(buffer.data['obs'], expected_data['obs'])
    assert torch.allclose(buffer.data['act'], expected_data['act'])
    assert torch.allclose(buffer.data['reward'], expected_data['reward'])
    assert torch.allclose(buffer.data['cost'], expected_data['cost'])
    assert torch.allclose(buffer.data['done'], expected_data['done'])
    assert torch.allclose(buffer.data['next_obs'], expected_data['next_obs'])

    # check the sample function
    data = buffer.sample_batch()
    assert data['obs'].shape == (batch_size * num_envs, *obs_space.shape)
    assert data['act'].shape == (batch_size * num_envs, *act_space.shape)
    assert data['reward'].shape == (batch_size * num_envs,)
    assert data['cost'].shape == (batch_size * num_envs,)
    assert data['done'].shape == (batch_size * num_envs,)
    assert data['next_obs'].shape == (batch_size * num_envs, *obs_space.shape)


@helpers.parametrize(
    obs_space=[Box(low=-1, high=1, shape=(1,))],
    act_space=[Box(low=-1, high=1, shape=(1,))],
    size=[10],
    batch_size=[5],
    device=[torch.device('cpu')],
)
def test_offpolicy_buffer(
    obs_space: Box,
    act_space: Box,
    size: int,
    batch_size: int,
    device: str,
) -> None:
    """Test buffer."""
    # initialize buffer
    buffer = OffPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=size,
        batch_size=batch_size,
        device=device,
    )
    # check the initialized value
    assert buffer.max_size == size, f'buffer.max_size is {buffer.max_size}'
    assert buffer.batch_size == batch_size, f'buffer.batch_size is {buffer.batch_size}'
    assert buffer.device == device, f'buffer.device is {buffer.device}'

    expected_data = {
        'obs': torch.randn((size, *obs_space.shape), dtype=torch.float32, device=device),
        'act': torch.randn((size, *act_space.shape), dtype=torch.float32, device=device),
        'reward': torch.randn((size), dtype=torch.float32, device=device),
        'cost': torch.randn((size), dtype=torch.float32, device=device),
        'done': torch.randn((size), dtype=torch.float32, device=device),
        'next_obs': torch.randn((size, *obs_space.shape), dtype=torch.float32, device=device),
    }
    # check the store function
    for i in range(size):
        buffer.store(
            obs=expected_data['obs'][i],
            act=expected_data['act'][i],
            reward=expected_data['reward'][i],
            cost=expected_data['cost'][i],
            done=expected_data['done'][i],
            next_obs=expected_data['next_obs'][i],
        )

    assert torch.allclose(buffer.data['obs'], expected_data['obs'])
    assert torch.allclose(buffer.data['act'], expected_data['act'])
    assert torch.allclose(buffer.data['reward'], expected_data['reward'])
    assert torch.allclose(buffer.data['cost'], expected_data['cost'])
    assert torch.allclose(buffer.data['done'], expected_data['done'])
    assert torch.allclose(buffer.data['next_obs'], expected_data['next_obs'])

    # check the sample function
    data = buffer.sample_batch()
    assert data['obs'].shape == (batch_size, *obs_space.shape)
    assert data['act'].shape == (batch_size, *act_space.shape)
    assert data['reward'].shape == (batch_size,)
    assert data['cost'].shape == (batch_size,)
    assert data['done'].shape == (batch_size,)
    assert data['next_obs'].shape == (batch_size, *obs_space.shape)
