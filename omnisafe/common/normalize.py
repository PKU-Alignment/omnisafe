import numpy as np
import torch.nn as nn
import torch

class Normalize(nn.Module):
    """
    Calculate normalized input from running mean and std
    See https://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self, shape, clip=1e6):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(*shape), requires_grad=False) # Current value of data stream
        self.mean = nn.Parameter(torch.zeros(*shape), requires_grad=False) # Current mean
        self.sumsq = nn.Parameter(torch.zeros(*shape), requires_grad=False) # Current sum of squares, used in var/std calculation

        self.var = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current variance
        self.std = nn.Parameter(torch.zeros(*shape), requires_grad=False)  # Current std

        self.count = nn.Parameter(torch.zeros(1), requires_grad=False) # Counter

        self.clip = nn.Parameter(clip*torch.ones(*shape), requires_grad=False) 

    def push(self, x):
        self.x.data = x
        self.count.data[0] += 1
        if self.count.data[0] == 1:
            self.mean.data = x
        else:
            old_mean = self.mean
            self.mean.data += (x - self.mean.data) / self.count.data
            self.sumsq.data += (x - old_mean.data) * (x - self.mean.data)
            self.var.data = self.sumsq.data / (self.count.data-1)
            self.std.data = torch.sqrt(self.var.data)
            self.std.data = torch.max(self.std.data, 1e-2*torch.ones_like(self.std.data))

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def get_std(self):
        return self.std
    
    def pre_process(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return x

    def normalize(self, x=None):
        x = self.pre_process(x)
        if x is not None:
            self.push(x)
            if self.count <= 1:
                return self.x.data
            else:
                output= (self.x.data - self.mean.data) / self.std.data
        else:
            output = (self.x - self.mean) / self.std
        return torch.clamp(output, -self.clip.data, self.clip.data)