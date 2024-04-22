# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""OmniSafe tools package."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn
import yaml
from rich.console import Console

from omnisafe.typing import DEVICE_CPU


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened parameters from the model.

    .. note::
        Some algorithms need to get the flattened parameters from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the parameters are flattened
        and then used to calculate the loss.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> get_flat_params_from(model)
        tensor([1., 2., 3., 4.])

    Args:
        model (torch.nn.Module): model to be flattened.

    Returns:
        Flattened parameters.

    Raises:
        AssertionError: If no gradients were found in model parameters.
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
        Some algorithms need to get the flattened gradients from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the gradients are flattened
        and then used to calculate the loss.

    Args:
        model (torch.nn.Module): The model to be flattened.

    Returns:
        Flattened gradients.

    Raises:
        AssertionError: If no gradients were found in model parameters.
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
        Some algorithms (e.g. TRPO, CPO, etc.) need to set the parameters to the model, instead of
        using the ``optimizer.step()``.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> set_param_values_to_model(model, vals)
        >>> model.weight.data
        tensor([[1., 2.],
                [3., 4.]])

    Args:
        model (torch.nn.Module): The model to be set.
        vals (torch.Tensor): The parameters to be set.

    Raises:
        AssertionError: If the instance of the parameters is not ``torch.Tensor``, or the lengths of
            the parameters and the model parameters do not match.
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


def seed_all(seed: int) -> None:
    """This function is used to set the random seed for all the packages.

    .. hint::
        To reproduce the results, you need to set the random seed for all the packages. Including
        ``numpy``, ``random``, ``torch``, ``torch.cuda``, ``torch.backends.cudnn``.

    .. warning::
        If you want to use the ``torch.backends.cudnn.benchmark`` or ``torch.backends.cudnn.deterministic``
        and your ``cuda`` version is over 10.2, you need to set the ``CUBLAS_WORKSPACE_CONFIG`` and
        ``PYTHONHASHSEED`` environment variables.

    Args:
        seed (int): The random seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict.

    .. note::
        This function is used to convert the custom configurations to dict. For example, if the
        custom configurations are ``train_cfgs:use_wandb`` and ``True``, then the output dict will
        be ``{'train_cfgs': {'use_wandb': True}}``.

    Args:
        key_list (str): list of keys.
        value (Any): value.

    Returns:
        The converted dict.
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


def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> None:
    """Updater of multi-level dictionary.

    Args:
        total_dict (dict[str, Any]): The total dictionary.
        item_dict (dict[str, Any]): The item dictionary.

    Examples:
        >>> total_dict = {'a': {'b': 1, 'c': 2}}
        >>> item_dict = {'a': {'b': 3, 'd': 4}}
        >>> update_dict(total_dict, item_dict)
        >>> total_dict
        {'a': {'b': 3, 'c': 2, 'd': 4}}
    """
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


def load_yaml(path: str) -> dict[str, Any]:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name. Make sure
        your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        path (str): The path of the ``yaml`` file.

    Returns:
        The default kwargs.

    Raises:
        AssertionError: If the ``yaml`` file is not found.
    """
    with open(path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{path} error: {exc}') from exc

    return kwargs


def recursive_check_config(
    config: dict[str, Any],
    default_config: dict[str, Any],
    exclude_keys: tuple[str, ...] = (),
) -> None:
    """Check whether config is valid in default_config.

    Args:
        config (dict[str, Any]): The config to be checked.
        default_config (dict[str, Any]): The default config.
        exclude_keys (tuple of str, optional): The keys to be excluded. Defaults to ().

    Raises:
        AssertionError: If the type of the value is not the same as the default value.
        KeyError: If the key is not in default_config.
    """
    assert isinstance(config, dict), 'custom_cfgs must be a dict!'
    for key in config:
        if key not in default_config and key not in exclude_keys:
            raise KeyError(f'Invalid key: {key}')
        if config[key] is None:
            return
        if isinstance(config[key], dict) and key != 'env_cfgs':
            recursive_check_config(config[key], default_config[key])


def assert_with_exit(condition: bool, msg: str) -> None:
    """Assert with message.

    Examples:
        >>> assert_with_exit(1 == 2, '1 must equal to 2')
        AssertionError: 1 must equal to 2

    Args:
        condition (bool): condition to be checked.
        msg (str): message to be printed.

    Raises:
        AssertionError: If the condition is not satisfied.
    """
    try:
        assert condition
    except AssertionError:
        console = Console()
        console.print('ERROR: ' + msg, style='bold red')
        sys.exit(1)


def recursive_dict2json(dict_obj: dict[str, Any]) -> str:
    """This function is used to recursively convert the dict to json.

    Args:
        dict_obj (dict[str, Any]): dict to be converted.

    Returns:
        The converted json string.

    Raises:
        AssertionError: If the instance of the input is not ``dict``.
    """
    assert isinstance(dict_obj, dict), 'Input must be a dict.'
    flat_dict: dict[str, Any] = {}

    def _flatten_dict(dict_obj: dict[str, Any] | Any, path: str = '') -> None:
        if isinstance(dict_obj, dict):
            for key, value in dict_obj.items():
                _flatten_dict(value, path + key + ':')
        else:
            flat_dict[path[:-1]] = dict_obj

    _flatten_dict(dict_obj)
    return json.dumps(flat_dict, sort_keys=True).replace('"', "'")


def hash_string(string: str) -> str:
    """This function is used to generate the folder name.

    Args:
        string (str): string to be hashed.

    Returns:
        The hashed string.
    """
    salt = b'\xf8\x99/\xe4\xe6J\xd8d\x1a\x9b\x8b\x98\xa2\x1d\xff3*^\\\xb1\xc1:e\x11M=PW\x03\xa5\\h'
    # convert string to bytes and add salt
    salted_string = salt + string.encode('utf-8')
    # use sha256 to hash
    hash_object = hashlib.sha256(salted_string)
    # get the hex digest
    return hash_object.hexdigest()


def get_device(device: torch.device | str | int = DEVICE_CPU) -> torch.device:
    """Retrieve PyTorch device.

    It checks that the requested device is available first. For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    Args:
        device (torch.device, str, or int, optional): The device to use. Defaults to
            ``torch.device('cpu')``.

    Returns:
        The device to use.
    """
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if not torch.cuda.is_available() and device.type == torch.device('cuda').type:
        return torch.device('cpu')

    return device
