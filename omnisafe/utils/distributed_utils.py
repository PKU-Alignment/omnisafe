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
"""torch.distributed for multi-processing"""

import os
import subprocess
import sys
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def setup_torch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    old_num_threads = torch.get_num_threads()
    # decrease number of torch threads for MPI
    if old_num_threads > 1 and num_procs() > 1:
        fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
        torch.set_num_threads(fair_num_threads)
        print(
            f'Proc {proc_id()}: Decreased number of Torch threads from '
            f'{old_num_threads} to {torch.get_num_threads()}',
            flush=True,
        )


def mpi_avg_grads(module: torch.nn.Module) -> None:
    """Average contents of gradient buffers across MPI processes.

    Args:
        module (torch.nn.Module): module to be averaged.
    """
    if num_procs() > 1:
        for parameter in module.parameters():
            p_grad = parameter.grad  # numpy view of tensor data
            avg_p_grad = mpi_avg(parameter.grad)
            p_grad[:] = avg_p_grad[:]


def sync_params(module: torch.nn.Module) -> None:
    """Sync all parameters of module across all MPI processes.

    .. note::

        This function only works when the training is multi-processing.

    Args:
        module (torch.nn.Module): module to be synchronized.
    """
    if num_procs() > 1:
        for parameter in module.parameters():
            p_numpy = parameter.data
            broadcast(p_numpy)


def mpi_fork(
    parallel: int,
    bind_to_core: bool = False,
    use_number_of_threads: bool = False,
    device: str = 'cpu',
) -> bool:
    """The entrance of multi-processing.

    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name <https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py>`_.

    .. note::

        Usage: if ``mpi_fork(n)`` : ``sys.exit()``

    Args:
        parallel (int): number of processes to launch.
        bind_to_core (bool, optional): Defaults to False.
        use_number_of_threads (bool, optional): Defaults to False.
    """
    is_parent = False
    back_end = 'gloo' if device == 'cpu' else 'nccl'
    if os.getenv('MASTER_ADDR') is not None:
        dist.init_process_group(backend=back_end)
    # Check if MPI is already setup..
    if parallel > 1 and os.getenv('MASTER_ADDR') is None:
        # MPI is not yet set up: quit parent process and start N child processes
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
        if bind_to_core:
            args += ['-bind-to', 'core']
        if use_number_of_threads:
            args += ['--use-hwthread-cpus']
        args += sys.argv
        # This is the parent process, spawn sub-processes..
        subprocess.check_call(args, env=env)
        is_parent = True
    return is_parent


def is_root_process() -> bool:
    """Test whether the process is the root process."""
    return bool(dist.get_rank() == 0)


def proc_id() -> int:
    """Get rank of calling process."""
    if os.getenv('MASTER_ADDR') is None:
        return 0
    return dist.get_rank()


def allreduce(*args, **kwargs) -> torch.Tensor:
    """Allreduce operation."""
    return dist.all_reduce(*args, **kwargs)


def gather(*args, **kwargs) -> torch.Tensor:
    """Gather operation."""
    return dist.gather(*args, **kwargs)


def gather_and_stack(x_vector: np.ndarray) -> np.ndarray:
    """Gather values from all tasks and return a flattened list.

    Input is a 1D array of size ``N``, and output is a list of size ``N * MPI_world_size``.

    .. note::

        Only the root process owns valid data.

    Args:
        x: 1-D array of size N
    """
    if num_procs() == 1:
        return x_vector
    assert x_vector.ndim == 1, 'Only lists or 1D-arrays supported.'
    buf = None
    size = num_procs()
    length = x_vector.shape[0]
    # if proc_id() == 0:
    buf = np.empty([size, length], dtype=np.float32)
    gather(x_vector, buf, root=0)
    return buf.flatten()


def num_procs() -> int:
    """Count active MPI processes."""
    if os.getenv('MASTER_ADDR') is None:
        return 1
    return dist.get_world_size()


def broadcast(value: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast."""
    dist.broadcast(value, src=src)


def mpi_avg(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar or numpy vector over MPI processes."""
    return mpi_sum(value) / num_procs()


def mpi_max(value: torch.Tensor) -> torch.Tensor:
    """Determine global maximum of scalar or numpy array over MPI processes."""
    return mpi_op(value, ReduceOp.MAX)


def mpi_min(value: torch.Tensor) -> torch.Tensor:
    """Determine global minimum of scalar or numpy array over MPI processes."""
    return mpi_op(value, ReduceOp.MIN)


def mpi_op(value: torch.Tensor, operation: ReduceOp) -> torch.Tensor:
    """Multi-processing operation.

    .. note::

        The operation can be ``ReduceOp.SUM``, ``ReduceOp.MAX``, ``ReduceOp.MIN``.
        corresponding to :meth:`mpi_sum`, :meth:`mpi_max`, :meth:`mpi_min`, respectively.

    Args:
        value (torch.Tensor): value to be operated.
        operation (ReduceOp): operation type.
    """
    if num_procs() == 1:
        return value
    value, scalar = ([value], True) if np.isscalar(value) else (value, False)
    value = torch.as_tensor(value, dtype=torch.float32)
    allreduce(value, op=operation)
    return value[0] if scalar else value


def mpi_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum a scalar or numpy vector over MPI processes."""
    return mpi_op(value, ReduceOp.SUM)


def mpi_avg_torch_tensor(value: torch.Tensor) -> None:
    """Average a torch tensor over MPI processes.

    Since torch and numpy share same memory space,
    tensors of dim > 0 can be be manipulated through call by reference,
    scalars must be assigned.

    Args:
        value (torch.Tensor): value to be averaged.
    """
    assert isinstance(value, torch.Tensor)
    if num_procs() > 1:
        # tensors can be manipulated in-place
        if len(value.shape) > 0:
            # x_numpy = x.numpy()  # numpy view of tensor data
            avg_x = mpi_avg(value)
            value[:] = avg_x[:]  # in-place memory update
        else:
            # scalars (tensors of dim = 0) must be assigned
            raise NotImplementedError


def mpi_statistics_scalar(
    value: torch.Tensor, with_min_and_max: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        value (torch.Tensor): value to be operated.
        with_min_and_max (bool): whether to return min and max.
    """
    # value = np.array(value, dtype=np.float32)
    global_sum, global_n = mpi_sum([torch.sum(value), len(value)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(torch.sum((value - mean) ** 2))
    # compute global std
    std = torch.sqrt(global_sum_sq / global_n)
    if with_min_and_max:
        global_min = mpi_op(np.min(value) if len(value) > 0 else np.inf, operation=ReduceOp.MIN)
        global_max = mpi_op(np.max(value) if len(value) > 0 else -np.inf, operation=ReduceOp.MAX)
        return mean, std, global_min, global_max
    return mean, std
