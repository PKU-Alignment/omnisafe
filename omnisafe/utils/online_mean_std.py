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
"""Implementation of the online mean and standard deviation."""

import numpy as np
import torch

from omnisafe.utils import distributed_utils


class OnlineMeanStd(torch.nn.Module):
    """
    Track mean and standard deviation of inputs with incremental formula.
    """

    def __init__(self, epsilon=1e-5, shape=()):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(*shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eps = epsilon
        self.bound = 10
        self.shape = shape

    @property
    def var(self):
        """Return variance."""
        return torch.square(self.std)

    @staticmethod
    def _convert_to_torch(params, dtype=torch.float32) -> torch.Tensor:
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        if isinstance(params, float):
            params = torch.tensor([params], dtype=dtype)  # use [] to make tensor torch.Size([1])
        if isinstance(params, np.floating):
            params = torch.tensor([params], dtype=dtype)  # use [] to make tensor torch.Size([1])
        return params

    def forward(self, data, subtract_mean=True, clip=False):
        """Make input average free and scale to standard deviation."""
        # sanity checks
        if len(data.shape) >= 2:
            assert (
                data.shape[-1] == self.mean.shape[-1]
            ), f'got shape={data.shape} but expected: {self.mean.shape}'

        is_numpy = isinstance(data, np.ndarray)
        data = self._convert_to_torch(data)
        if subtract_mean:
            data_new = (data - self.mean) / (self.std + self.eps)
        else:
            data_new = data / (self.std + self.eps)
        if clip:
            data_new = torch.clamp(data_new, -self.bound, self.bound)
        data_new = data_new.numpy() if is_numpy else data_new
        return data_new

    # pylint: disable-next=too-many-locals
    def update(self, data) -> None:
        """Update internals incrementally.
        Note: works for both vector and matrix inputs.

        MPI implementation according to Chan et al.[10]; see:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        data = self._convert_to_torch(data)

        # ==== Input checks
        msg = f'Expected dim in [1, 2], but got dim={len(data.shape)}.'
        assert len(data.shape) == 2 or len(data.shape) == 1, msg
        if self.shape[0] > 1:  # expect matrix inputs
            msg = f'Expected obs_dim={self.shape[0]} but got: {data.shape[1]}'
            assert len(data.shape) == 2 and data.shape[1] == self.shape[0], msg
        if self.shape[0] == 1:
            assert len(data.shape) == 1, f'Expected dim=1 but got: {data.shape}'
            # reshape is necessary since mean operator reduces vector dim by one
            data = data.view((-1, 1))

        n_b = data.shape[0] * distributed_utils.num_procs()  # get batch size
        n_a = self.count.clone()
        n_a_b = self.count + n_b
        batch_mean = torch.mean(data, dim=0)

        # 1) Calculate mean and average batch mean across processes
        distributed_utils.mpi_avg_torch_tensor(batch_mean)
        delta = batch_mean - self.mean
        mean_new = self.mean + delta * n_b / n_a_b

        # 2) Determine variance and sync across processes
        diff = data - mean_new
        batch_var = torch.mean(diff**2, dim=0)
        distributed_utils.mpi_avg_torch_tensor(batch_var)

        # Update running terms
        m2_a = n_a * self.var
        m2_b = n_b * batch_var
        ratio = n_a * n_b / n_a_b
        m2_a_b = m2_a + m2_b + delta**2 * ratio

        # 3) Update parameters - access internal values with data attribute
        self.mean.data = mean_new
        self.count.data = n_a_b
        new_var = m2_a_b / n_a_b
        self.std.data = torch.sqrt(new_var)
