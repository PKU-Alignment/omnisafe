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

from typing import Tuple, Union

import numpy as np
import scipy.signal
import torch

from omnisafe.models import ConstraintActorCritic, ConstraintActorQCritic


# global dict that holds pointers to functions
registered_actors = {}


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


def register_actor(actor_name: str) -> torch.nn.Module:
    """Register actor into global dict.

    Args:
        actor_name (str): actor name.
    """

    def wrapper(func):
        registered_actors[actor_name] = func
        return func

    return wrapper


def get_registered_actor_fn(actor_type: str, distribution_type: str) -> torch.nn.Module:
    """Get registered actor function.

    Args:
        actor_type (str): actor type.
        distribution_type (str): distribution type.
    """
    assert distribution_type in ('categorical', 'gaussian')
    actor_fn = actor_type + '_' + distribution_type
    msg = f'Did not find: {actor_fn} in registered actors.'
    assert actor_fn in registered_actors, msg
    return registered_actors[actor_fn]


def combined_shape(
    length: int, shape: tuple = None
) -> Union[Tuple[int,], Tuple[int, int]]:
    """Combined vectors shape.

    This function is used to combine vector shape.

    - If shape is None, return (length,).
    - If shape is a scalar, return (length, shape).
    - If shape is a tuple, return (length, shape[0], shape[1], ...).

    Args:
        length (int): length of vectors.
        shape (tuple): shape of vectors.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: torch.nn.Module) -> float:
    """Count variables in module.

    .. warning::
        It is recommended to use this function to count variables in module,
        instead of using ``sum([np.prod(p.shape) for p in module.parameters()])``.
        For more details, please refer to the following link:
        `pylint request
        <https://pylint.pycqa.org/en/latest/user_guide/messages/refactor/consider-using-generator.html>`_

    Args:
        module (torch.nn.Module): module to be counted.
    """
    # https://pylint.pycqa.org/en/latest/user_guide/messages/refactor/consider-using-generator.html
    # Don't use sum([np.prod(p.shape) for p in module.parameters()])
    return sum(np.prod(p.shape) for p in module.parameters())


def discount_cumsum(x_vector: np.ndarray, discount: float) -> np.ndarray:
    r"""Magic from RLlib for computing discounted cumulative sums of vectors.

    For example:
    The input is :math:`x = [x_0, x_1, x_2]`, and the output is
    :math:`y = [x_0 + \gamma  x_1 + \gamma^2  x_2, x_1 + \gamma  x_2, x_2]`.
    where :math:`\gamma` is the discount factor.

    Args:
        x_vector (np.ndarray): input vector.
        discount (float): discount factor.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x_vector[::-1], axis=0)[::-1]
