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
"""Some Core Functions"""

from typing import Union

import torch

from omnisafe.models import ConstraintActorCritic, ConstraintActorQCritic


def set_optimizer(
    opt: str, module: Union[ConstraintActorCritic, ConstraintActorQCritic], learning_rate: float
) -> torch.optim.Optimizer:
    """Returns an initialized optimizer from PyTorch.

    .. note::

        The optimizer can be chosen from the following list:

        - Adam
        - AdamW
        - Adadelta
        - Adagrad
        - Adamax
        - ASGD
        - LBFGS
        - RMSprop
        - Rprop
        - SGD

    Args:
        opt (str): optimizer name.
        module (torch.nn.Module): module to be optimized.
        learning_rate (float): learning rate.
    """
    assert hasattr(torch.optim, opt), f'Optimizer={opt} not found in torch.'
    optimizer = getattr(torch.optim, opt)

    return optimizer(module.parameters(), lr=learning_rate, eps=1e-5)


def discount_cumsum_torch(x_vector: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors."""
    length = x_vector.shape[0]
    x_vector = x_vector.type(torch.float64)
    for idx in reversed(range(length)):
        if idx == length - 1:
            cumsum = x_vector[idx]
        else:
            cumsum = x_vector[idx] + discount * cumsum
        x_vector[idx] = cumsum
    return x_vector
