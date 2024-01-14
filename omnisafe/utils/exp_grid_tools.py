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
"""Tools for Experiment Grid."""


from __future__ import annotations

import os
import string
import sys
from typing import Any

import omnisafe
from omnisafe.typing import Tuple


def all_bools(vals: list[Any]) -> bool:
    """Check if all values are bools.

    Args:
        vals (list[Any]): Values to check.

    Returns:
        True if all values are bools, False otherwise.
    """
    return all(isinstance(v, bool) for v in vals)


def valid_str(vals: list[Any] | str) -> str:
    """Convert a value or values to a string which could go in a path of file.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    Args:
        vals (list[Any] or str): Values to convert.

    Returns:
        Converted string.
    """
    if isinstance(vals, (list, tuple)):
        return '-'.join([valid_str(x) for x in vals])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(vals).lower()
    valid_chars = f'-_{string.ascii_letters}{string.digits}'
    return ''.join(c if c in valid_chars else '-' for c in str_v)


def train(
    exp_id: str,
    algo: str,
    env_id: str,
    custom_cfgs: dict[str, Any],
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (Config): Custom configurations.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    with open(
        os.path.join(
            f'{custom_cfgs["logger_cfgs"]["log_dir"]}',
            terminal_log_name,
        ),
        'w',
        encoding='utf-8',
    ) as f_out:
        sys.stdout = f_out
        with open(
            os.path.join(
                f'{custom_cfgs["logger_cfgs"]["log_dir"]}',
                error_log_name,
            ),
            'w',
            encoding='utf-8',
        ) as f_error:
            sys.stderr = f_error
            agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
            reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len
