# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import yaml
from rich.console import Console
from torch.version import cuda as cuda_version

from omnisafe.typing import cpu


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened parameters from the model.

    .. note::
        Some algorithms need to get the flattened parameters from the model,
        such as the :class:`TRPO` and :class:`CPO` algorithm.
        In these algorithms, the parameters are flattened and then used to calculate the loss.

    Example:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> get_flat_params_from(model)
        tensor([1., 2., 3., 4.])

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


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    """This function is used to set the parameters to the model.

    .. note::
        Some algorithms (e.g. TRPO, CPO, etc.) need to set the parameters to the model,
        instead of using the ``optimizer.step()``.

    Example:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> set_param_values_to_model(model, vals)
        >>> model.weight.data
        tensor([[1., 2.],
                [3., 4.]])

    Args:
        model (torch.nn.Module): model to be set.
        vals (torch.Tensor): parameters to be set.
    """
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'


def seed_all(seed: int):
    """This function is used to set the random seed for all the packages.

    .. hint::
        To reproduce the results, you need to set the random seed for all the packages.
        Including ``numpy``, ``random``, ``torch``, ``torch.cuda``, ``torch.backends.cudnn``.

    .. warning::
        If you want to use the ``torch.backends.cudnn.benchmark`` or ``torch.backends.cudnn.
        deterministic`` and your ``cuda`` version is over 10.2, you need to set the
        ``CUBLAS_WORKSPACE_CONFIG`` and ``PYTHONHASHSEED`` environment variables.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        if cuda_version is not None and float(cuda_version) >= 10.2:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
    except AttributeError:
        pass


def custom_cfgs_to_dict(key_list, value):
    """This function is used to convert the custom configurations to dict.

    .. note::
        This function is used to convert the custom configurations to dict.
        For example, if the custom configurations are ``train_cfgs:use_wandb`` and ``True``,
        then the output dict will be ``{'train_cfgs': {'use_wandb': True}}``.

    Args:
        key_list (list): list of keys.
        value: value.
    """
    if value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif '.' in value:
        value = float(value)
    elif value.isdigit():
        value = int(value)
    elif value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
        value = value.split(',')
    else:
        value = str(value)
    keys_split = key_list.replace('-', '_').split(':')
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace('-', '_'): return_dict}
    return return_dict


def update_dict(total_dict, item_dict):
    """Updater of multi-level dictionary."""
    for idd in item_dict:
        total_value = total_dict.get(idd)
        item_value = item_dict.get(idd)

        if total_value is None:
            total_dict.update({idd: item_value})
        elif isinstance(item_value, dict):
            update_dict(total_value, item_value)
            total_dict.update({idd: total_value})
        else:
            total_value = item_value
            total_dict.update({idd: total_value})


def load_yaml(path) -> dict:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name.
        Make sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        path (str): path of the ``yaml`` file.
    """
    with open(path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
        except yaml.YAMLError as exc:
            raise AssertionError(f'{path} error: {exc}') from exc

    return kwargs


def recursive_check_config(config, default_config, exclude_keys=()):
    """Check whether config is valid in default_config.

    Args:
        config (dict): config to be checked.
        default_config (dict): default config.
    """
    for key in config:
        if key not in default_config.keys() and key not in exclude_keys:
            raise KeyError(f'Invalid key: {key}')
        if isinstance(config[key], dict):
            recursive_check_config(config[key], default_config[key])


def assert_with_exit(condition, msg) -> None:
    """Assert with message.

    Args:
        condition (bool): condition to be checked.
        msg (str): message to be printed.
    """
    try:
        assert condition
    except AssertionError:
        console = Console()
        console.print('ERROR: ' + msg, style='bold red')
        sys.exit(1)


def recursive_dict2json(dict_obj) -> str:
    """This function is used to recursively convert the dict to json.

    Args:
        dict_obj (dict): dict to be converted.
    """
    assert isinstance(dict_obj, dict), 'Input must be a dict.'
    flat_dict = {}

    def _flatten_dict(dict_obj, path=''):
        if isinstance(dict_obj, dict):
            for key, value in dict_obj.items():
                _flatten_dict(value, path + key + ':')
        else:
            flat_dict[path[:-1]] = dict_obj

    _flatten_dict(dict_obj)
    return json.dumps(flat_dict, sort_keys=True).replace('"', "'")


def hash_string(string) -> str:
    """This function is used to generate the folder name.

    Args:
        string (str): string to be hashed.
    """
    salt = b'\xf8\x99/\xe4\xe6J\xd8d\x1a\x9b\x8b\x98\xa2\x1d\xff3*^\\\xb1\xc1:e\x11M=PW\x03\xa5\\h'
    # convert string to bytes and add salt
    salted_string = salt + string.encode('utf-8')
    # use sha256 to hash
    hash_object = hashlib.sha256(salted_string)
    # get the hex digest
    return hash_object.hexdigest()


def get_device(device: torch.device = cpu) -> torch.device:
    """Retrieve PyTorch device.

    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    Args:
        device (torch.device): device to be used.

    Returns:
        torch.device: device to be used.
    """
    # Cuda by default
    if device == 'auto':
        device = 'cuda'
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device('cuda').type and not torch.cuda.is_available():
        return torch.device('cpu')

    return device
