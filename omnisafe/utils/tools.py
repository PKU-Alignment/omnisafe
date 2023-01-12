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
"""tool_function_packages"""

import os

import numpy as np
import torch
import yaml

from omnisafe.typing import Any, Callable, Dict, Union


def get_default_kwargs_yaml(algo: str, env_id: str, algo_type: str) -> Dict:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name.
        Make sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        algo (str): algorithm name.
        env_id (str): environment name.
        algo_type (str): algorithm type.
    """
    path = os.path.abspath(__file__).split('/')[:-2]
    cfg_path = os.path.join('/', *path, 'configs', algo_type, f'{algo}.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f'{algo}.yaml error: {exc}'
    kwargs_name = env_id if env_id in kwargs.keys() else 'defaults'
    return kwargs[kwargs_name]


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened parameters from the model.

    .. note::
        Some algorithms need to get the flattened parameters from the model,
        such as the :class:`TRPO` and :class:`CPO` algorithm.
        In these algorithms, the parameters are flattened and then used to calculate the loss.

    Args:
        model (torch.nn.Module): model to be flattened.
    """
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, 'No gradients were found in model parameters.'
    return torch.cat(flat_params)


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened gradients from the model.

    .. note::
        Some algorithms need to get the flattened gradients from the model,
        such as the :class:`TRPO` and :class:`CPO` algorithm.
        In these algorithms, the gradients are flattened and then used to calculate the loss.

    Args:
        model (torch.nn.Module): model to be flattened.
    """
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, 'No gradients were found in model parameters.'
    return torch.cat(grads)


def conjugate_gradients(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b_vector: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
):  # pylint: disable=invalid-name,too-many-locals

    """Implementation of Conjugate gradient algorithm.

    Conjugate gradient algorithm is used to solve the linear system of equations :math:`Ax = b`.
    The algorithm is described in detail in the paper `Conjugate Gradient Method`_.

    .. _Conjugate Gradient Method: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        Increasing ``num_steps`` will lead to a more accurate approximation
        to :math:`A^{-1} b`, and possibly slightly-improved performance,
        but at the cost of slowing things down.
        Also probably don't play with this hyperparameter.

    Args:
        num_steps (int): Number of iterations of conjugate gradient to perform.
    """

    x = torch.zeros_like(b_vector)
    r = b_vector - Avp(x)
    p = r.clone()
    rdotr = torch.dot(r, r)

    for _ in range(num_steps):
        z = Avp(p)
        alpha = rdotr / (torch.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr
    return x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    """This function is used to set the parameters to the model.

    .. note::
        Some algorithms (e.g. TRPO, CPO, etc.) need to set the parameters to the model,
        instead of using the ``optimizer.step()``.

    Args:
        model (torch.nn.Module): model to be set.
        vals (torch.Tensor): parameters to be set.
    """
    assert isinstance(vals, torch.Tensor)
    i = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : i + size]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += size  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'


# pylint: disable-next=too-many-branches,too-many-return-statements
def to_ndarray(item: Any, dtype: np.dtype = None) -> Union[np.ndarray, TypeError, None]:
    """This function is used to convert the data type to ndarray.

    Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.

    .. note:
        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

    Args:
        item (Any): item to be converted.
        dtype (np.dtype): data type of the output ndarray. Default to None.
    """

    if isinstance(item, dict):
        new_data = {}
        for key, value in item.items():
            new_data[key] = to_ndarray(value, dtype)
        return new_data

    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return None
        if hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        new_data = []
        for data in item:
            new_data.append(to_ndarray(data, dtype))
        return new_data

    if isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        return item.numpy().astype(dtype)

    if isinstance(item, np.ndarray):
        if dtype is None:
            return item
        return item.astype(dtype)

    if isinstance(item, (bool, str)):
        return item

    if np.isscalar(item):
        return np.array(item)

    if item is None:
        return None

    raise TypeError(f'not support item type: {item}')


def expand_dims(*args):
    """This function is used to expand the dimensions of the input data.

    .. note::
        This function is used to expand the dimensions of the input data.
        For example, if the input data is a scalar, then the output data will be a 1-dim ndarray.

    Args:
        *args: input data to be expanded.
    """
    if len(args) == 1:
        return np.expand_dims(args[0], axis=0)
    return [np.expand_dims(item, axis=0) for item in args]


def as_tensor(*args):
    """This function is used to convert the input data to tensor.

    .. note::
        This function is used to convert the input data to tensor.
        For example, if the input data is a scalar, then the output data will be a 0-dim tensor.

    Args:
        *args: input data to be converted.
    """
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=torch.float32)
    return [torch.as_tensor(item, dtype=torch.float32) for item in args]
