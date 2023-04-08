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
"""Implementation of the Logger."""

from __future__ import annotations

import atexit
import csv
import os
import time
from collections import deque
from typing import Any, Deque, TextIO

import numpy as np
import torch
import tqdm
import wandb
from rich import print  # pylint: disable=redefined-builtin
from rich.console import Console
from rich.table import Table

from omnisafe.utils.config import Config
from omnisafe.utils.distributed import dist_statistics_scalar, get_rank


# As of torch v1.9.0, torch.utils.tensorboard has a bug that is exposed by setuptools 59.6.0.  The
# bug is that it attempts to import distutils then access distutils.version without actually
# importing distutils.version.  We can workaround this by prepopulating the distutils.version
# submodule in the distutils module.

try:  # noqa: SIM105
    # pylint: disable-next=wrong-import-order,unused-import,deprecated-module
    import distutils.version  # isort:skip  # noqa: F401
except ImportError:
    pass

# pylint: disable-next=wrong-import-order
from torch.utils.tensorboard.writer import SummaryWriter  # isort:skip


class Logger:  # pylint: disable=too-many-instance-attributes
    """Implementation of the Logger.

    A logger to record the training process.
    It can record the training process to a file and print it to the console.
    It can also record the training process to tensorboard.

    The logger can record the following data:

    .. code-block:: bash

        ----------------------------------------------
        |       Name      |            Value         |
        ----------------------------------------------
        |    Train/Epoch  |             25           |
        |  Metrics/EpCost |            24.56         |
        |  Metrics/EpLen  |            1000          |
        |  Metrics/EpRet  |            13.24         |
        |  Metrics/EpStd  |            0.12          |
        ----------------------------------------------
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
        config: Config | None = None,
        models: list[torch.nn.Module] | None = None,
    ) -> None:
        """Initialize the logger.

        Args:
            output_dir: The directory to save the log file.
            exp_name: The name of the experiment.
            output_fname: The name of the log file.
            verbose: Whether to print the log to the console.
            seed: The random seed.
            use_tensorboard: Whether to use tensorboard.
            use_wandb: Whether to use wandb.
            config: The config of the experiment.
            models: The models to be saved.
        """
        hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        relpath = hms_time

        if seed is not None:
            relpath = f'seed-{str(seed).zfill(3)}-{relpath}'

        self._hms_time = hms_time
        self._log_dir = os.path.join(output_dir, exp_name, relpath)
        self._verbose = verbose
        self._maste_proc = get_rank() == 0
        self._console = Console()

        self._output_file: TextIO
        if self._maste_proc:
            os.makedirs(self._log_dir, exist_ok=True)
            self._output_file = open(  # noqa: SIM115 # pylint: disable=consider-using-with
                os.path.join(self._log_dir, output_fname),
                encoding='utf-8',
                mode='w',
            )
            atexit.register(self._output_file.close)
            self.log(f'Logging data to {self._output_file.name}', 'cyan', bold=True)
            self._csv_writer = csv.writer(self._output_file)

        self._epoch: int = 0
        self._first_row: bool = True
        self._what_to_save: dict[str, Any] | None = None
        self._data: dict[str, Deque[int | float] | list[int | float]] = {}
        self._headers_windwos: dict[str, int | None] = {}
        self._headers_minmax: dict[str, bool] = {}
        self._headers_delta: dict[str, bool] = {}
        self._current_row: dict[str, int | float] = {}

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

    def log(self, msg: str, color: str = 'green', bold: bool = False) -> None:
        """Log the message to the console and the file.

        Args:
            msg (str): The message to be logged.
            color (int): The color of the message.
            bold (bool): Whether to use bold font.
        """
        if self._verbose and self._maste_proc:
            style = ' '.join([color, 'bold' if bold else ''])
            self._console.print(msg, style=style)

    def save_config(self, config: Config) -> None:
        """Save the configuration to the log directory.

        Args:
            config (Config): The configuration to be saved.
        """
        if self._maste_proc:
            self.log('Save with config in config.json', 'yellow', bold=True)
            with open(os.path.join(self._log_dir, 'config.json'), encoding='utf-8', mode='w') as f:
                f.write(config.tojson())

    def setup_torch_saver(self, what_to_save: dict[str, Any]) -> None:
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
        window_length: int | None = None,
        min_and_max: bool = False,
        delta: bool = False,
    ) -> None:
        """Register a key to the logger.

        The logger can record the following data:

        .. code-block:: bash

            ----------------------------------------------------
            |       Name            |            Value         |
            ----------------------------------------------------
            |    Train/Epoch        |             25           |
            |  Metrics/EpCost/Min   |            22.38         |
            |  Metrics/EpCost/Max   |            25.48         |
            |  Metrics/EpCost/Mean  |            23.93         |
            ----------------------------------------------------

        Args:
            key (str): The key to be registered.
            window_length (int): The window length for the key, \
                if window_length is None, the key will be averaged in epoch.
            min_and_max (bool): Whether to record the min and max value of the key.
            delta (bool): Whether to record the delta value of the key.
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

    def store(self, **kwargs: int | float | np.ndarray | torch.Tensor) -> None:
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
        """Dump the tabular data to the console and the file.

        The dumped data will be separated by the following steps:

        .. hint::

            - If the key is registered with window_length, the data will be averaged in the window.
            - Write the data to the csv file.
            - Write the data to the tensorboard.
            - Update the progress logger.

        """
        self._update_current_row()
        table = Table('Metrics', 'Value')
        if self._maste_proc:
            self._epoch += 1
            if self._verbose:
                key_lens = list(map(len, self._current_row.keys()))
                max_key_len = max(15, *key_lens)
                for key, val in self._current_row.items():
                    table.add_row(key[:max_key_len], str(val)[:max_key_len])

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
        self._console.print(table)

    def _update_current_row(self) -> None:
        """Update the current row.

        Update the current row with the data stored in the logger.
        """
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

    def get_stats(self, key, min_and_max: bool = False) -> tuple[int | float, ...]:
        """Get the statistics of the key.

        Args:
            key (str): The key to be registered.
            min_and_max (bool): Whether to record the min and max value of the key.
        """
        assert key in self._current_row, f'Key {key} has not been registered'
        vals = self._data[key]
        if isinstance(vals, deque):
            vals = list(vals)

        if min_and_max:
            mean, std, min_val, max_val = dist_statistics_scalar(
                torch.tensor(vals),
                with_min_and_max=True,
            )
            return mean.item(), min_val.item(), max_val.item(), std.item()

        mean, std = dist_statistics_scalar(  # pylint: disable=unbalanced-tuple-unpacking
            torch.tensor(vals),
        )
        return (mean.item(),)

    @property
    def current_epoch(self) -> int:
        """Return the current epoch."""
        return self._epoch

    @property
    def log_dir(self) -> str:
        """Return the log directory."""
        return self._log_dir

    def close(self) -> None:
        """Close the logger."""
        if self._maste_proc:
            self._output_file.close()
