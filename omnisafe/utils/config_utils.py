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
"""config_utils"""

from collections import OrderedDict, namedtuple
from typing import Any, Dict, NamedTuple, Union


def recursive_update(args: dict, update_args: dict, add_new_args: bool = False) -> NamedTuple:
    """Recursively update args.

    This function is used to update the args from :class:`dict` to :class:`namedtuple`.
    If you want to update your own args, you can use this function.

    Args:
        args (dict): args to be updated.
        update_args (dict): args to update.
    """
    if update_args is not None:
        for key, value in args.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    print(f'{key}:')
                    recursive_update(args[key], update_args[key])
                else:
                    args[key] = update_args[key]
                    menus = (key, update_args[key])
                    print(f'- {menus[0]}: {menus[1]} is update!')
                if isinstance(value, dict):
                    recursive_update(value, update_args)
    if add_new_args:
        for key, value in update_args.items():
            if key not in args:
                args[key] = value
                menus = (key, value)
                print(f'- {menus[0]}: {menus[1]} is added!')

    return dict2namedtuple(args)


def dict2namedtuple(obj: Any) -> Union[NamedTuple, Dict, Any]:
    """Create namedtuple from dict.

    This function is used to convert the args from :class:`dict` to :class:`namedtuple`.

    Args:
        obj (dict, tuple, set, frozenset): objects to be converted.
    """
    if isinstance(obj, dict):
        fields = sorted(obj.keys())
        namedtuple_type = namedtuple('GenericObject', fields, rename=True)
        field_value_pairs = OrderedDict(
            (str(field), dict2namedtuple(obj[field])) for field in fields
        )
        try:
            return namedtuple_type(**field_value_pairs)
        except TypeError:
            # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
            return dict(**field_value_pairs)
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [dict2namedtuple(item) for item in obj]
    else:
        return obj


def namedtuple2dict(obj: Any) -> Union[Any, Dict[str, Any]]:
    """Create a dict from a namedtuple.

    This function is used to convert the args from :class:`namedtuple` to :class:`dict`.

    .. note::
        The function :meth:`_asdict` of :class:`namedtuple` is not recursive,
        so we need to implement this function to save the args.

    Args:
        obj (dict, tuple, set, frozenset): objects to be converted.
    """
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return {key: namedtuple2dict(value) for key, value in obj._asdict().items()}
    return obj


def check_all_configs(configs: NamedTuple, algo_type: str) -> None:
    """Check all configs.

    This function is used to check the configs.

    .. note::

        For on-policy algorithms.

        - pi_iters and critic_iters must be greater than 0.
        - actor_lr and critic_lr must be greater than 0.
        - gamma must be in [0, 1).
        - if use_cost is False, cost_gamma must be 1.0.

        For off-policy algorithms.

        - actor_lr and critic_lr must be greater than 0.
        - replay_buffer size must be greater than batch_size.
        - update_every must be less than steps_per_epoch.

    Args:
        configs (dict): configs to be checked.
        algo_type (str): algorithm type.
    """
    if algo_type == 'on-policy':
        assert (
            configs.actor_iters > 0 and configs.critic_iters > 0
        ), 'pi_iters and critic_iters must be greater than 0'
        assert (
            configs.actor_lr > 0 and configs.critic_lr > 0
        ), 'actor_lr and critic_lr must be greater than 0'
        assert (
            configs.buffer_cfgs.gamma >= 0 and configs.buffer_cfgs.gamma < 1.0
        ), 'gamma must be in [0, 1)'
        assert (
            configs.use_cost is False and configs.cost_gamma == 1.0
        ) or configs.use_cost, 'if use_cost is False, cost_gamma must be 1.0'
    elif algo_type == 'off-policy':
        assert (
            configs.actor_lr > 0 and configs.critic_lr > 0
        ), 'actor_lr and critic_lr must be greater than 0'
        assert (
            configs.replay_buffer_cfgs.size > configs.replay_buffer_cfgs.batch_size
        ), 'replay_buffer size must be greater than batch_size'
        assert (
            configs.update_every < configs.steps_per_epoch
        ), 'update_every must be less than steps_per_epoch'
