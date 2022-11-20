"""tool_function_packages"""
import os
from typing import Any

import numpy as np
import torch
import yaml


# pylint: disable=E1101


# def get_defaults_kwargs_yaml_on_policy(algo, env_id):
#     """get_defaults_kwargs_yaml_on_policy"""
#     path = os.path.abspath(__file__).split('/')[:-2]
#     cfg_path = os.path.join('/', *path, 'configs/on_policy_cfgs', f'{algo}.yaml')
#     with open(cfg_path, 'r', encoding='utf-8') as file:
#         try:
#             kwargs = yaml.load(file, Loader=yaml.FullLoader)
#         except yaml.YAMLError as exc:
#             assert False, f'{algo}.yaml error: {exc}'
#     kwargs_name = env_id if env_id in kwargs.keys() else 'defaults'
#     return kwargs[kwargs_name]


# def get_defaults_kwargs_yaml_off_policy(algo, env_id=None):
#     """get_defaults_kwargs_yaml_off_policy"""
#     path = os.path.abspath(__file__).split('/')[:-2]
#     cfg_path = os.path.join('/', *path, 'configs/off_policy_cfgs', f'{algo}.yaml')
#     with open(cfg_path, 'r', encoding='utf-8') as file:
#         try:
#             kwargs = yaml.load(file, Loader=yaml.FullLoader)
#         except yaml.YAMLError as exc:
#             assert False, f'{algo}.yaml error: {exc}'
#     kwargs_name = env_id if env_id in kwargs.keys() else 'defaults'
#     return kwargs[kwargs_name]


def get_default_kwargs_yaml(algo, env_id, on_policy=True):
    """get_default_kwargs_yaml"""
    path = os.path.abspath(__file__).split('/')[:-2]
    dir_name = 'on_policy_cfgs' if on_policy else 'off_policy_cfgs'
    cfg_path = os.path.join('/', *path, 'configs', dir_name, f'{algo}.yaml')
    print(cfg_path)
    with open(cfg_path, 'r', encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f'{algo}.yaml error: {exc}'
    kwargs_name = env_id if env_id in kwargs.keys() else 'defaults'
    return kwargs[kwargs_name]


def get_flat_params_from(model):
    """get_flat_params_from"""
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, 'No gradients were found in model parameters.'
    # pylint: disable=E1101
    return torch.cat(flat_params)


def get_flat_gradients_from(model):
    """get_flat_gradients_from"""
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, 'No gradients were found in model parameters.'

    return torch.cat(grads)


# pylint: disable=R0914,C0103
def conjugate_gradients(Avp, b_vector, nsteps, residual_tol=1e-10, eps=1e-6):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)

    nsteps: (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.
    """
    # pylint disable=E1101
    x = torch.zeros_like(b_vector)
    r = b_vector - Avp(x)
    p = r.clone()
    rdotr = torch.dot(r, r)

    fmtstr = '%10i %10.3g %10.3g'
    verbose = False

    for i in range(nsteps):
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
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


def set_param_values_to_model(model, vals):
    """set_param_values_to_model"""
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


def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError('not support item type: {}'.format(type(item)))
