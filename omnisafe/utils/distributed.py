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
"""Tools of torch.distributed for multi-processing."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def setup_distributed() -> None:
    """Setup the distributed training environment.

    Avoid slowdowns caused by each separate process's PyTorch, using more than its fair share of CPU
    resources.
    """
    old_num_threads = torch.get_num_threads()
    # decrease number of torch threads for MPI
    if old_num_threads > 1 and world_size() > 1:
        fair_num_threads = max(int(torch.get_num_threads() / world_size()), 1)
        torch.set_num_threads(fair_num_threads)
        print(
            f'Proc {get_rank()}: Decreased number of Torch threads from '
            f'{old_num_threads} to {torch.get_num_threads()}',
            flush=True,
        )


def get_rank() -> int:
    """Get the rank of calling process.

    Examples:
        >>> # In process 0
        >>> get_rank()
        0

    Returns:
        The rank of calling process.
    """
    if os.getenv('MASTER_ADDR') is None:
        return 0
    return dist.get_rank()


def world_size() -> int:
    """Count active MPI processes.

    Returns:
        The number of active MPI processes.
    """
    if os.getenv('MASTER_ADDR') is None:
        return 1
    return dist.get_world_size()


reduce = dist.reduce
all_reduce = dist.all_reduce
gather = dist.gather
all_gather = dist.all_gather
broadcast = dist.broadcast
scatter = dist.scatter


def fork(
    parallel: int,
    device: str = 'cpu',
    manual_args: list[str] | None = None,
) -> bool:
    """The entrance method of multi-processing.

    Re-launches the current script with workers linked by MPI. Also, terminates the original process
    that launched it. Taken almost without modification from the Baselines function of the
    `same name <https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py>`_.

    Args:
        parallel (int): The number of processes to launch.
        device (str, optional): The device to be used. Defaults to 'cpu'.
        manual_args (list of str or None, optional): The arguments to be passed to the new
            processes. Defaults to None.
    """
    backend = 'gloo' if device == 'cpu' else 'nccl'
    if os.getenv('MASTER_ADDR') is not None and os.getenv('IN_DIST') is None:
        dist.init_process_group(backend=backend)
        os.environ['IN_DIST'] = '1'
    # check if MPI is already setup..
    if parallel > 1 and os.getenv('MASTER_ADDR') is None:
        # MPI is not yet set up: quit parent process and start N child processes
        if device != 'cpu':
            initial_device = int(device.split(':')[-1])
            os.environ['USE_DISTRIBUTED'] = '1'
            if os.getenv('CUDA_VISIBLE_DEVICES') is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                    str(initial_device + i) for i in range(parallel)
                )
            num_gpu = int((len(os.environ['CUDA_VISIBLE_DEVICES']) + 1) / 2)
            assert (
                num_gpu >= parallel
            ), f'Please make sure you have enough available GPUs to run Parallel {parallel}, \
                current available Devices are {num_gpu}.'
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', IN_MPI='1')
        args = [
            'torchrun',
            '--rdzv_backend',
            'c10d',
            '--rdzv_endpoint',
            'localhost:0',
            '--nproc_per_node',
            str(parallel),
        ]
        if manual_args is not None:
            args += manual_args
            print(manual_args)
        else:
            args += sys.argv
            print(sys.argv)
        # this is the parent process, spawn sub-processes..
        subprocess.check_call(args, env=env)  # noqa: S603
        return True
    return False


def avg_tensor(value: torch.Tensor) -> None:
    """Average a torch tensor over MPI processes.

    Since torch and numpy share same memory space, tensors of dim > 0 can be be manipulated through
    call by reference, scalars must be assigned.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> avg_tensor(x)
        >>> x
        tensor(1.5)

    Args:
        value (torch.Tensor): The value to be averaged.
    """
    assert isinstance(value, torch.Tensor)
    if world_size() > 1:
        assert len(value.shape) > 0
        avg_x = dist_avg(value)
        value[:] = avg_x[:]


def avg_grads(module: torch.nn.Module) -> None:
    """Average contents of gradient buffers across MPI processes.

    .. note::
        This function only works when the training is multi-processing.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0, requires_grad=True)
        >>> y = x ** 2
        >>> y.backward()
        >>> x.grad
        tensor(2.)
        >>> # In process 1
        >>> x = torch.tensor(2.0, requires_grad=True)
        >>> y = x ** 2
        >>> y.backward()
        >>> x.grad
        tensor(4.)
        >>> avg_grads(x)
        >>> x.grad
        tensor(3.)

    Args:
        module (torch.nn.Module): The module in which grad need to be averaged.
    """
    if world_size() > 1:
        for parameter in module.parameters():
            if parameter.grad is not None:
                p_grad = parameter.grad
                avg_p_grad = dist_avg(parameter.grad)
                p_grad[:] = avg_p_grad[:]


def sync_params(module: torch.nn.Module) -> None:
    """Sync all parameters of module across all MPI processes.

    .. note::
        This function only works when the training is multi-processing.

    Examples:
        >>> # In process 0
        >>> model = torch.nn.Linear(1, 1)
        >>> model.weight.data = torch.tensor([[1.]])
        >>> model.weight.data
        tensor([[1.]])
        >>> # In process 1
        >>> model = torch.nn.Linear(1, 1)
        >>> model.weight.data = torch.tensor([[2.]])
        >>> model.weight.data
        tensor([[2.]])
        >>> sync_params(model)
        >>> model.weight.data
        tensor([[1.]])

    Args:
        module (torch.nn.Module): The module to be synchronized.
    """
    if world_size() > 1:
        for parameter in module.parameters():
            p_numpy = parameter.data
            broadcast(p_numpy, src=0)


def avg_params(module: torch.nn.Module) -> None:
    """Average contents of all parameters across MPI processes.

    Examples:
        >>> # In process 0
        >>> model = torch.nn.Linear(1, 1)
        >>> model.weight.data = torch.tensor([[1.]])
        >>> model.weight.data
        tensor([[1.]])
        >>> # In process 1
        >>> model = torch.nn.Linear(1, 1)
        >>> model.weight.data = torch.tensor([[2.]])
        >>> model.weight.data
        tensor([[2.]])
        >>> avg_params(model)
        >>> model.weight.data
        tensor([[1.5]])

    Args:
        module (torch.nn.Module): The module in which parameters need to be averaged.
    """
    if world_size() > 1:
        for parameter in module.parameters():
            param_tensor = parameter.data
            avg_param_tensor = dist_avg(param_tensor)
            param_tensor[:] = avg_param_tensor[:]


def dist_avg(value: np.ndarray | torch.Tensor | float) -> torch.Tensor:
    """Average a tensor over distributed processes.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> dist_avg(x)
        tensor(1.5)

    Args:
        value (np.ndarray, torch.Tensor, int, or float): value to be averaged.

    Returns:
        Averaged tensor.
    """
    return dist_sum(value) / world_size()


def dist_max(value: np.ndarray | torch.Tensor | float) -> torch.Tensor:
    """Determine global maximum of tensor over distributed processes.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> dist_max(x)
        tensor(2.)

    Args:
        value (np.ndarray, torch.Tensor, int, or float): value to be find max value.

    Returns:
        Maximum tensor.
    """
    return dist_op(value, ReduceOp.MAX)


def dist_min(value: np.ndarray | torch.Tensor | float) -> torch.Tensor:
    """Determine global minimum of tensor over distributed processes.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> dist_min(x)
        tensor(1.)

    Args:
        value (np.ndarray, torch.Tensor, int, or float): value to be find min value.

    Returns:
        Minimum tensor.
    """
    return dist_op(value, ReduceOp.MIN)


def dist_sum(value: np.ndarray | torch.Tensor | float) -> torch.Tensor:
    """Sum a tensor over distributed processes.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> dist_sum(x)
        tensor(3.)

    Args:
        value (np.ndarray, torch.Tensor, int, or float): The value to be summed.

    Returns:
        Summed tensor.
    """
    return dist_op(value, ReduceOp.SUM)


def dist_op(value: np.ndarray | torch.Tensor | float, operation: Any) -> torch.Tensor:
    """Multi-processing operation.

    .. note::
        The operation can be ``ReduceOp.SUM``, ``ReduceOp.MAX``, ``ReduceOp.MIN``. corresponding to
        :meth:`dist_sum`, :meth:`dist_max`, :meth:`dist_min`, respectively.

    Args:
        value (np.ndarray, torch.Tensor, int, or float): The value to be operated.
        operation (ReduceOp): operation type.

    Returns:
        Operated (SUM, MAX, MIN) tensor.
    """
    if world_size() == 1:
        return torch.as_tensor(value, dtype=torch.float32)
    value_, scalar = ([value], True) if np.isscalar(value) else (value, False)
    value = torch.as_tensor(value_, dtype=torch.float32)
    all_reduce(value, op=operation)
    return value[0] if scalar else value


def dist_statistics_scalar(
    value: torch.Tensor,
    with_min_and_max: bool = False,
) -> tuple[torch.Tensor, ...]:
    r"""Get mean/std and optional min/max of scalar x across MPI processes.

    Examples:
        >>> # In process 0
        >>> x = torch.tensor(1.0)
        >>> # In process 1
        >>> x = torch.tensor(2.0)
        >>> dist_statistics_scalar(x)
        (tensor(1.5), tensor(0.5))

    Args:
        value (torch.Tensor): Value to be operated.
        with_min_and_max (bool, optional): whether to return min and max. Defaults to False.

    Returns:
        A tuple of the [mean, std] or [mean, std, min, max] of the input tensor.
    """
    global_sum = dist_sum(torch.sum(value))
    global_n = dist_sum(torch.tensor(len(value)).to(os.getenv('OMNISAFE_DEVICE', 'cpu')))
    mean = global_sum / global_n

    global_sum_sq = dist_sum(torch.sum((value - mean) ** 2))
    # compute global std
    std = torch.sqrt(global_sum_sq / global_n)
    if with_min_and_max:
        global_min = dist_min(value)
        global_max = dist_max(value)
        return mean, std, global_min, global_max
    return mean, std
