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

import os
import random

import numpy as np
import torch


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


def as_tensor(*args, device: torch.device = torch.device('cpu')):
    """This function is used to convert the input data to tensor.

    .. note::
        This function is used to convert the input data to tensor.
        For example, if the input data is a scalar, then the output data will be a 0-dim tensor.

    Args:
        *args: input data to be converted.
    """
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=torch.float32)
    return [torch.as_tensor(item, dtype=torch.float32, device=device) for item in args]


def seed_all(seed: int):
    """This function is used to set the random seed for all the packages."""

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
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


def update_dic(total_dic, item_dic):
    '''Updater of multi-level dictionary.'''
    for idd in item_dic.keys():
        total_value = total_dic.get(idd)
        item_value = item_dic.get(idd)

        if total_value is None:
            total_dic.update({idd: item_value})
        elif isinstance(item_value, dict):
            update_dic(total_value, item_value)
            total_dic.update({idd: total_value})
        else:
            total_value = item_value
            total_dic.update({idd: total_value})


if __name__ == '__main__':
    print('This is a tool function package.')
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', 'True'))
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', 'False'))
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', '0.1'))
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', '1'))
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', 'test'))
    print(custom_cfgs_to_dict('train_cfgs:use_wandb', '[1,2,3]'))
