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
"""Test Buffers"""

import torch

import helpers
from omnisafe.common.buffer import Buffer
from omnisafe.common.vector_buffer import VectorBuffer


@helpers.parametrize(
    obs_dim=[1],
    act_dim=[1],
    size=[100],
    gamma=[0.9],
    lam=[0.9],
    adv_estimation_method=['gae', 'vtrace', 'gae-rtg', 'plain'],
    standardized_rew_adv=[True],
    standardized_cost_adv=[True],
    lam_c=[0.9],
    penalty_param=[0.0],
    device=['cpu'],
    num_envs=[1],
)
def test_vector_buffer(
    obs_dim: int,
    act_dim: int,
    size: int,
    gamma: float,
    lam: float,
    adv_estimation_method: str,
    standardized_rew_adv: bool,
    standardized_cost_adv: bool,
    lam_c: float,
    penalty_param: float,
    device: str,
    num_envs: int,
) -> None:
    """Test the VectorBuffer class."""
    vector_buffer = VectorBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=size,
        gamma=gamma,
        lam=lam,
        adv_estimation_method=adv_estimation_method,
        standardized_rew_adv=standardized_rew_adv,
        standardized_cost_adv=standardized_cost_adv,
        lam_c=lam_c,
        penalty_param=penalty_param,
        device=device,
        num_envs=num_envs,
    )
    # checking the initialized value
    assert (
        vector_buffer.num_buffers == num_envs
    ), f'vector_buffer.num_buffers is {vector_buffer.num_buffers}'
    assert (
        vector_buffer.standardized_cost_adv == standardized_cost_adv
    ), f'vector_buffer.standardized_cost_adv is {vector_buffer.standardized_cost_adv}'
    assert (
        vector_buffer.standardized_rew_adv == standardized_rew_adv
    ), f'vector_buffer.standardized_rew_adv is {vector_buffer.standardized_rew_adv}'
    assert vector_buffer.buffers is not [], f'vector_buffer.buffers is {vector_buffer.buffers}'

    # checking the store function
    for _ in range(size):
        obs = torch.rand((num_envs, obs_dim), dtype=torch.float32, device=device)
        act = torch.rand((num_envs, act_dim), dtype=torch.float32, device=device)
        rew = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        cost = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        val = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        cost_val = torch.rand((num_envs, 1), dtype=torch.float32, device=device)
        log_p = torch.rand((num_envs, 1), dtype=torch.float32, device=device)

        vector_buffer.store(obs, act, rew, val, log_p, cost, cost_val)
        for idx, buffer in enumerate(vector_buffer.buffers):
            assert torch.allclose(
                buffer.obs_buf[buffer.ptr - 1], obs[idx]
            ), f'buffer.obs[idx] is {buffer.obs_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.act_buf[buffer.ptr - 1], act[idx]
            ), f'buffer.act[idx] is {buffer.act_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.rew_buf[buffer.ptr - 1], rew[idx]
            ), f'buffer.rew[idx] is {buffer.rew_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.cost_buf[buffer.ptr - 1], cost[idx]
            ), f'buffer.cost[idx] is {buffer.cost_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.val_buf[buffer.ptr - 1], val[idx]
            ), f'buffer.val[idx] is {buffer.val_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.cost_val_buf[buffer.ptr - 1], cost_val[idx]
            ), f'buffer.cost_val[idx] is {buffer.cost_val_buf[buffer.ptr-1]}'
            assert torch.allclose(
                buffer.logp_buf[buffer.ptr - 1], log_p[idx]
            ), f'buffer.log_prob[idx] is {buffer.logp_buf[buffer.ptr-1]}'

    # checking the finish_path function
    for idx, buffer in enumerate(vector_buffer.buffers):
        last_val = torch.randn(1, device=device)
        last_cost_val = torch.randn(1, device=device)
        vector_buffer.finish_path(last_val, last_cost_val, idx)
        assert (
            buffer.path_start_idx == buffer.ptr
        ), f'buffer.path_start_idx is {buffer.path_start_idx}'

    # checking the get function
    for idx, buffer in enumerate(vector_buffer.buffers):
        data = vector_buffer.get()
        assert torch.allclose(
            data['obs'][idx : size * (idx + 1)], buffer.obs_buf
        ), f'data[obs][idx] is {data["obs"][idx]}'
        assert torch.allclose(
            data['act'][idx : size * (idx + 1)], buffer.act_buf
        ), f'data[act][idx] is {data["act"][idx]}'
        assert torch.allclose(
            data['log_p'][idx : size * (idx + 1)], buffer.logp_buf
        ), f'data[log_p][idx] is {data["log_p"][idx]}'


@helpers.parametrize(
    obs_dim=[5],
    act_dim=[5],
    size=[100],
    gamma=[0.9],
    lam=[0.9],
    adv_estimation_method=['gae', 'vtrace', 'gae-rtg', 'plain'],
    lam_c=[0.9],
    penalty_param=[0.0, 1.0],
    device=['cpu'],
)
def test_buffer(
    obs_dim: int,
    act_dim: int,
    size: int,
    gamma: float,
    lam: float,
    adv_estimation_method: str,
    lam_c: float,
    penalty_param: float,
    device: str,
) -> None:
    """Test buffer"""
    # initialize buffer
    buffer = Buffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=size,
        gamma=gamma,
        lam=lam,
        adv_estimation_method=adv_estimation_method,
        lam_c=lam_c,
        penalty_param=penalty_param,
        device=device,
    )
    # checking the initialized value
    assert buffer.ptr == 0, f'buffer.ptr is {buffer.ptr}'
    assert buffer.path_start_idx == 0, f'buffer.path_start_idx is {buffer.path_start_idx}'
    assert buffer.size == size, f'buffer.size is {buffer.size}'
    assert buffer.max_size == size, f'buffer.max_size is {buffer.max_size}'
    assert buffer.gamma == gamma, f'buffer.gamma is {buffer.gamma}'
    assert buffer.lam == lam, f'buffer.lam is {buffer.lam}'
    assert buffer.lam_c == lam_c, f'buffer.lam_c is {buffer.lam_c}'
    assert buffer.penalty_param == penalty_param, f'buffer.penalty_param is {buffer.penalty_param}'
    assert (
        buffer.adv_estimation_method == adv_estimation_method
    ), f'buffer.adv_estimation_method is {buffer.adv_estimation_method}'
    assert buffer.device == device, f'buffer.device is {buffer.device}'
    expected_data = {
        'obs': torch.zeros((size, obs_dim), dtype=torch.float32, device=device),
        'act': torch.zeros((size, act_dim), dtype=torch.float32, device=device),
        'log_p': torch.zeros((size), dtype=torch.float32, device=device),
    }
    # check the store function
    for _ in range(size):
        obs, act, rew, val, logp, cost, cost_val = (
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
        expected_data['log_p'][buffer.ptr] = logp
        buffer.store(obs, act, rew, val, logp, cost, cost_val)
        assert torch.allclose(
            buffer.obs_buf[buffer.ptr - 1], obs
        ), f'buffer.obs_buf[buffer.ptr-1] is {buffer.obs_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.act_buf[buffer.ptr - 1], act
        ), f'buffer.act_buf[buffer.ptr-1] is {buffer.act_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.rew_buf[buffer.ptr - 1], rew
        ), f'buffer.rew_buf[buffer.ptr-1] is {buffer.rew_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.val_buf[buffer.ptr - 1], val
        ), f'buffer.val_buf[buffer.ptr-1] is {buffer.val_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.logp_buf[buffer.ptr - 1], logp
        ), f'buffer.logp_buf[buffer.ptr-1] is {buffer.logp_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.cost_buf[buffer.ptr - 1], cost
        ), f'buffer.cost_buf[buffer.ptr-1] is {buffer.cost_buf[buffer.ptr-1]}'
        assert torch.allclose(
            buffer.cost_val_buf[buffer.ptr - 1], cost_val
        ), f'buffer.cost_val_buf[buffer.ptr-1] is {buffer.cost_val_buf[buffer.ptr-1]}'

    assert buffer.ptr == size, f'buffer.ptr is {buffer.ptr}'

    # check the finish_path function
    last_val = torch.randn(1, device=device)
    last_cost_val = torch.randn(1, device=device)
    buffer.finish_path(last_val=last_val, last_cost_val=last_cost_val)
    assert buffer.path_start_idx == buffer.ptr, f'buffer.path_start_idx is {buffer.path_start_idx}'

    # check the get function
    data = buffer.get()
    assert torch.allclose(data['obs'], expected_data['obs'])
    assert torch.allclose(data['act'], expected_data['act'])
    assert torch.allclose(data['log_p'], expected_data['log_p'])
    assert buffer.ptr == 0, f'buffer.ptr is {buffer.ptr}'
    assert buffer.path_start_idx == 0, f'buffer.path_start_idx is {buffer.path_start_idx}'
