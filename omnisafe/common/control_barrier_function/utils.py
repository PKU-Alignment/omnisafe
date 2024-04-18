import torch
import wrapt
import numpy as np
from typing import Collection, Callable, Mapping
import torch.nn as nn
import os
import requests
from torch import load

class Normalizer(nn.Module):
    def __init__(self, dim, *, clip=10):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('n', torch.tensor(0, dtype=torch.int64))
        self.placeholder = nn.Parameter(torch.tensor(0.), False)  # for device info (@maybe_numpy)
        self.clip = clip

    def forward(self, x, inverse=False):
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / self.std.clamp(min=1e-6)

    def update(self, data):
        data = data - self.mean

        m = data.shape[0]
        delta = data.mean(dim=0)
        new_n = self.n + m
        new_mean = self.mean + delta * m / new_n
        new_std = torch.sqrt((self.std**2 * self.n + data.var(dim=0) * m + delta**2 * self.n * m / new_n) / new_n)

        self.mean.set_(new_mean.data)
        self.std.set_(new_std.data)
        self.n.set_(new_n.data)

    def fit(self, data):
        n = data.shape[0]
        self.n.set_(torch.tensor(n, device=self.n.device))
        self.mean.set_(data.mean(dim=0))
        self.std.set_(data.std(dim=0))


def download_model(url, destination):
    response = requests.get(url)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        f.write(response.content)

def get_pretrained_model(model_path, model_url, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading from cloud...")
        download_model(model_url, model_path)
    else:
        print("Model found locally.")

    model = load(model_path, map_location=device)
    return model
