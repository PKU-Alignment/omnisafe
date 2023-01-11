import numpy as np
import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Calculate normalized input from running mean and std
    See https://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, shape, clip=1e6):
        """Initialize the normalize."""
        super().__init__()
        self.input = nn.Parameter(
            torch.zeros(*shape), requires_grad=False
        )  # Current value of data stream
        self.mean = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current mean
        self.sumsq = nn.Parameter(
            torch.zeros(*shape), requires_grad=False
        )  # Current sum of squares, used in var/std calculation

        self.var = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current variance
        self.std = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current std

        self.count = nn.Parameter(torch.zeros(1), requires_grad=False)  # Counter

        self.clip = nn.Parameter(clip * torch.ones(*shape), requires_grad=False)

    def push(self, input):
        """Push a new value into the stream."""
        self.input.data = input
        self.count.data[0] += 1
        if self.count.data[0] == 1:
            self.mean.data = input
        else:
            old_mean = self.mean
            self.mean.data += (input - self.mean.data) / self.count.data
            self.sumsq.data += (input - old_mean.data) * (input - self.mean.data)
            self.var.data = self.sumsq.data / (self.count.data - 1)
            self.std.data = torch.sqrt(self.var.data)
            self.std.data = torch.mainput(self.std.data, 1e-2 * torch.ones_like(self.std.data))

    def get_mean(self):
        """Get the mean value."""
        return self.mean

    def get_var(self):
        """Get the variance."""
        return self.var

    def get_std(self):
        """Get the std."""
        return self.std

    def pre_process(self, input):
        """Pre-process the input."""
        if isinstance(input, np.ndarray):
            input = torch.as_tensor(input, dtype=torch.float32)
        if len(input.shape) == 1:
            input = input.unsqueeze(-1)
        return input

    def normalize(self, input=None):
        """Nomalize the input."""
        input = self.pre_process(input)
        if input is not None:
            self.push(input)
            if self.count <= 1:
                return self.input.data
            output = (self.input.data - self.mean.data) / self.std.data
        else:
            output = (self.input - self.mean) / self.std
        return torch.clamp(output, -self.clip.data, self.clip.data)
