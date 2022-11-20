import numpy as np
import torch

import omnisafe.algos.utils.distributed_tools as distributed_tools


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
        return torch.square(self.std)

    @staticmethod
    def _convert_to_torch(x, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(x, float):
            x = torch.tensor([x], dtype=dtype)  # use [] to make tensor torch.Size([1])
        if isinstance(x, np.floating):
            x = torch.tensor([x], dtype=dtype)  # use [] to make tensor torch.Size([1])
        return x

    def forward(self, x, subtract_mean=True, clip=False):
        """Make input average free and scale to standard deviation."""
        # sanity checks
        if len(x.shape) >= 2:
            assert (
                x.shape[-1] == self.mean.shape[-1]
            ), f'got shape={x.shape} but expected: {self.mean.shape}'

        is_numpy = isinstance(x, np.ndarray)
        x = self._convert_to_torch(x)
        if subtract_mean:
            x_new = (x - self.mean) / (self.std + self.eps)
        else:
            x_new = x / (self.std + self.eps)
        if clip:
            x_new = torch.clamp(x_new, -self.bound, self.bound)
        x_new = x_new.numpy() if is_numpy else x_new
        return x_new

    def update(self, x) -> None:
        """Update internals incrementally.
        Note: works for both vector and matrix inputs.

        MPI implementation according to Chan et al.[10]; see:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        x = self._convert_to_torch(x)

        # ==== Input checks
        msg = f'Expected dim in [1, 2], but got dim={len(x.shape)}.'
        assert len(x.shape) == 2 or len(x.shape) == 1, msg
        if self.shape[0] > 1:  # expect matrix inputs
            msg = f'Expected obs_dim={self.shape[0]} but got: {x.shape[1]}'
            assert len(x.shape) == 2 and x.shape[1] == self.shape[0], msg
        if self.shape[0] == 1:
            assert len(x.shape) == 1, f'Expected dim=1 but got: {x.shape}'
            # reshape is necessary since mean operator reduces vector dim by one
            x = x.view((-1, 1))

        n_B = x.shape[0] * distributed_tools.num_procs()  # get batch size
        n_A = self.count.clone()
        n_AB = self.count + n_B
        batch_mean = torch.mean(x, dim=0)

        # 1) Calculate mean and average batch mean across processes
        distributed_tools.mpi_avg_torch_tensor(batch_mean)
        delta = batch_mean - self.mean
        mean_new = self.mean + delta * n_B / n_AB

        # 2) Determine variance and sync across processes
        diff = x - mean_new
        batch_var = torch.mean(diff**2, dim=0)
        distributed_tools.mpi_avg_torch_tensor(batch_var)

        # Update running terms
        M2_A = n_A * self.var
        M2_B = n_B * batch_var
        ratio = n_A * n_B / n_AB
        M2_AB = M2_A + M2_B + delta**2 * ratio

        # 3) Update parameters - access internal values with data attribute
        self.mean.data = mean_new
        self.count.data = n_AB
        new_var = M2_AB / n_AB
        self.std.data = torch.sqrt(new_var)
