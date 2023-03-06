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
"""Implementation of the Logger."""

import atexit
import csv
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Literal, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import tqdm
import wandb

from omnisafe.utils.config import Config
from omnisafe.utils.distributed import dist_statistics_scalar, get_rank


# As of torch v1.9.0, torch.utils.tensorboard has a bug that is exposed by setuptools 59.6.0.  The
# bug is that it attempts to import distutils then access distutils.version without actually
# importing distutils.version.  We can workaround this by prepopulating the distutils.version
# submodule in the distutils module.

try:
    # pylint: disable-next=wrong-import-order,unused-import
    import distutils.version  # isort:skip  # noqa: F401
except ImportError:
    pass

# pylint: disable-next=wrong-import-order
from torch.utils.tensorboard import SummaryWriter  # isort:skip


ColorType = Literal['gray', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'crimson']


class WordColor:  # pylint: disable=too-few-public-methods
    """Implementation of the WordColor."""

    GRAY: int = 30
    RED: int = 31
    GREEN: int = 32
    YELLOW: int = 33
    BLUE: int = 34
    MAGENTA: int = 35
    CYAN: int = 36
    WHITE: int = 37
    CRIMSON: int = 38

    @staticmethod
    def colorize(msg: str, color: str, bold: bool = False, highlight: bool = False) -> str:
        """Colorize a message.

        Args:
            msg (str): message to be colorized.
            color (str): color of the message.
            bold (bool): whether to use bold font.
            highlight (bool): whether to use highlight.

        Returns:
            str: colorized message.
        """
        assert color.upper() in WordColor.__dict__, f'Invalid color: {color}'
        color_code = WordColor.__dict__[color.upper()]
        attr = []
        if highlight:
            color_code += 10
        attr.append(str(color_code))
        if bold:
            attr.append('1')
        return f'\x1b[{";".join(attr)}m{msg}\x1b[0m'


class Logger:  # pylint: disable=too-many-instance-attributes
    """Implementation of the Logger.

    A logger to record the training process.
    It can record the training process to a file and print it to the console.
    It can also record the training process to tensorboard.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        output_dir: str,
        exp_name: str,
        output_fname: str = 'progress.csv',
        verbose: bool = True,
        seed: int = 0,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Optional[Config] = None,
        models: Optional[List[torch.nn.Module]] = None,
    ) -> None:
        hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        relpath = hms_time

        if seed is not None:
            relpath = f'seed-{str(seed).zfill(3)}-{relpath}'

        self._hms_time = hms_time
        self._log_dir = os.path.join(output_dir, exp_name, relpath)
        self._verbose = verbose
        self._maste_proc = get_rank() == 0

        self._output_file: TextIO
        if self._maste_proc:
            os.makedirs(self._log_dir, exist_ok=True)
            self._output_file = open(  # pylint: disable=consider-using-with
                os.path.join(self._log_dir, output_fname), encoding='utf-8', mode='w'
            )
            atexit.register(self._output_file.close)
            self.log(f'Logging data to {self._output_file.name}', 'cyan', bold=True)
            self._csv_writer = csv.writer(self._output_file)

        self._epoch: int = 0
        self._first_row: bool = True
        self._what_to_save: Optional[Dict[str, Any]] = None
        self._data: Dict[str, Union[Deque[Union[int, float]], List[Union[int, float]]]] = {}
        self._headers_windwos: Dict[str, Optional[int]] = {}
        self._headers_minmax: Dict[str, bool] = {}
        self._headers_delta: Dict[str, bool] = {}
        self._current_row: Dict[str, Union[int, float]] = {}

        if config is not None:
            self.save_config(config)
            self._config = config

        self._use_tensorboard = use_tensorboard
        self._use_wandb = use_wandb

        if self._use_tensorboard and self._maste_proc:
            self._tensorboard_writer = SummaryWriter(log_dir=os.path.join(self._log_dir, 'tb'))

        if self._use_wandb and self._maste_proc:
            project: str = self._config.logger_cfgs.get('wandb_project', 'omnisafe')
            name: str = f'{exp_name}-{relpath}'
            print('project', project, 'name', name)
            wandb.init(project=project, name=name, dir=self._log_dir, config=config)
            if config is not None:
                wandb.config.update(config)
            if models is not None:
                for model in models:
                    wandb.watch(model)

        if not self._verbose:
            assert (
                'epochs' in self._config
            ), 'epochs must be specified in the config file when verbose is False'
            self._proc_bar = tqdm.tqdm(total=self._config['epochs'], desc='Epochs')

    def log(
        self, msg: str, color: ColorType = 'green', bold: bool = False, highlight: bool = False
    ) -> None:
        """Log the message to the console and the file.

        Args:
            msg (str): The message to be logged.
            color (int): The color of the message.
        """
        if self._verbose and self._maste_proc:
            print(WordColor.colorize(msg, color, bold, highlight))

    def save_config(self, config: Config) -> None:
        """Save the configuration to the log directory.

        Args:
            config (dict): The configuration to be saved.
        """
        if self._maste_proc:
            self.log('Save with config in config.json', 'yellow', bold=True)
            with open(os.path.join(self._log_dir, 'config.json'), encoding='utf-8', mode='w') as f:
                f.write(config.tojson())

    def setup_torch_saver(self, what_to_save: Dict[str, Any]) -> None:
        """Setup the torch saver.

        Args:
            what_to_save (dict): The dict of the things to be saved.
        """
        self._what_to_save = what_to_save

    def torch_save(self) -> None:
        """Save the torch model."""
        if self._maste_proc:
            assert self._what_to_save is not None, 'Please setup torch saver first'
            path = os.path.join(self._log_dir, 'torch_save', f'epoch-{self._epoch}.pt')
            os.makedirs(os.path.dirname(path), exist_ok=True)

            params = {
                k: v.state_dict() if hasattr(v, 'state_dict') else v
                for k, v in self._what_to_save.items()
            }
            torch.save(params, path)

    def register_key(
        self,
        key: str,
        window_length: Optional[int] = None,
        min_and_max: bool = False,
        delta: bool = False,
    ) -> None:
        """Register a key to the logger.

        Args:
            key (str): The key to be registered.
            window_length (int): The window length for the key, \
                if window_length is None, the key will be averaged in epoch.
        """
        assert (key and f'{key}/Mean') not in self._current_row, f'Key {key} has been registered'
        if min_and_max:
            self._current_row[f'{key}/Mean'] = 0
            self._current_row[f'{key}/Min'] = 0
            self._current_row[f'{key}/Max'] = 0
            self._current_row[f'{key}/Std'] = 0
            self._headers_minmax[key] = True

        else:
            self._current_row[key] = 0
            self._headers_minmax[key] = False

        if delta:
            self._current_row[f'{key}/Delta'] = 0
            self._headers_delta[key] = True
        else:
            self._headers_delta[key] = False

        if window_length is not None:
            self._data[key] = deque(maxlen=window_length)
            self._headers_windwos[key] = window_length
        else:
            self._data[key] = []
            self._headers_windwos[key] = None

    def store(self, **kwargs: Union[int, float, np.ndarray, torch.Tensor]) -> None:
        """Store the data to the logger.

        Args:
            **kwargs: The data to be stored.
        """
        for key, val in kwargs.items():
            assert key in self._current_row, f'Key {key} has not been registered'
            if isinstance(val, (int, float)):
                self._data[key].append(val)
            elif isinstance(val, torch.Tensor):
                self._data[key].append(val.mean().item())
            elif isinstance(val, np.ndarray):
                self._data[key].append(val.mean())
            else:
                raise ValueError(f'Unsupported type {type(val)}')

    def dump_tabular(self) -> None:
        """Dump the tabular data to the console and the file."""
        self._update_current_row()
        if self._maste_proc:
            self._epoch += 1
            if self._verbose:
                key_lens = list(map(len, self._current_row.keys()))
                max_key_len = max(15, *key_lens)
                n_slashes = 22 + max_key_len
                print('-' * n_slashes)
                for key, val in self._current_row.items():
                    print(f'| {key:<{max_key_len}} | {val:15.6g} |')
                print('-' * n_slashes)
            else:
                self._proc_bar.update(1)

            if self._first_row:
                self._csv_writer.writerow(self._current_row.keys())
                self._first_row = False
            self._csv_writer.writerow(self._current_row.values())
            self._output_file.flush()

            if self._use_tensorboard:
                for key, val in self._current_row.items():
                    self._tensorboard_writer.add_scalar(key, val, global_step=self._epoch)
                self._tensorboard_writer.flush()

            if self._use_wandb:
                wandb.log(self._current_row, step=self._epoch)

    def _update_current_row(self) -> None:
        for key in self._data:
            if self._headers_minmax[key]:
                old_data = self._current_row[f'{key}/Mean']
                mean, min_val, max_val, std = self.get_stats(key, True)
                self._current_row[f'{key}/Mean'] = mean
                self._current_row[f'{key}/Min'] = min_val
                self._current_row[f'{key}/Max'] = max_val
                self._current_row[f'{key}/Std'] = std
            else:
                old_data = self._current_row[key]
                mean = self.get_stats(key, False)[0]
                self._current_row[key] = mean

            if self._headers_delta[key]:
                self._current_row[f'{key}/Delta'] = mean - old_data

            if self._headers_windwos[key] is None:
                self._data[key] = []

    def get_stats(self, key, min_and_max: bool = False) -> Tuple[Union[int, float], ...]:
        """Get the statistics of the key."""
        assert key in self._current_row, f'Key {key} has not been registered'
        vals = self._data[key]
        if isinstance(vals, deque):
            vals = list(vals)

        if min_and_max:
            mean, std, min_val, max_val = dist_statistics_scalar(
                torch.tensor(vals), with_min_and_max=True
            )
            return mean.item(), min_val.item(), max_val.item(), std.item()

        mean, std = dist_statistics_scalar(  # pylint: disable=unbalanced-tuple-unpacking
            torch.tensor(vals)
        )
        return (mean.item(),)

    def close(self) -> None:
        """Close the logger."""
        if self._maste_proc:
            self._output_file.close()
