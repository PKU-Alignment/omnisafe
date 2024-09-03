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
from typing import Any, TextIO

import numpy as np
import torch
import wandb
from rich import print  # pylint: disable=redefined-builtin,wrong-import-order
from rich.console import Console  # pylint: disable=wrong-import-order
from rich.table import Table  # pylint: disable=wrong-import-order

from omnisafe.utils.config import Config
from omnisafe.utils.distributed import dist_statistics_scalar, get_rank


# As of torch v1.9.0, torch.utils.tensorboard has a bug that is exposed by setuptools 59.6.0.  The
# bug is that it attempts to import distutils then access distutils.version without actually
# importing distutils.version.  We can workaround this by prepopulating the distutils.version
# submodule in the distutils module.

try:  # noqa: SIM105
    # pylint: disable-next=wrong-import-order,unused-import,deprecated-module
    import distutils.version  # isort:skip  # noqa: F401
except ImportError:  # pragma: no cover
    pass

# pylint: disable-next=wrong-import-order
from torch.utils.tensorboard.writer import SummaryWriter  # isort:skip


class Logger:  # pylint: disable=too-many-instance-attributes
    """Implementation of the Logger.

    A logger to record the training process. It can record the training process to a file and print
    it to the console. It can also record the training process to tensorboard.

    The logger can record the following data:

    .. code-block:: text

        ----------------------------------------------
        |       Name      |            Value         |
        ----------------------------------------------
        |    Train/Epoch  |             25           |
        |  Metrics/EpCost |            24.56         |
        |  Metrics/EpLen  |            1000          |
        |  Metrics/EpRet  |            13.24         |
        |  Metrics/EpStd  |            0.12          |
        ----------------------------------------------

    Args:
        output_dir (str): The output directory.
        exp_name (str): The experiment name.
        output_fname (str, optional): The output file name. Defaults to 'progress.csv'.
        seed (int, optional): The random seed. Defaults to 0.
        use_tensorboard (bool, optional): Whether to use tensorboard. Defaults to True.
        use_wandb (bool, optional): Whether to use wandb. Defaults to False.
        config (Config or None, optional): The config. Defaults to None.
        models (list[torch.nn.Module] or None, optional): The models. Defaults to None.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        output_dir: str,
        exp_name: str,
        output_fname: str = 'progress.csv',
        seed: int = 0,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Config | None = None,
        models: list[torch.nn.Module] | None = None,
    ) -> None:
        """Initialize an instance of :class:`Logger`."""
        hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        relpath = hms_time

        if seed is not None:
            relpath = f'seed-{str(seed).zfill(3)}-{relpath}'

        self._hms_time: str = hms_time
        self._log_dir: str = os.path.join(output_dir, exp_name, relpath)
        self._maste_proc: bool = get_rank() == 0
        self._console: Console = Console()

        if self._maste_proc:
            os.makedirs(self._log_dir, exist_ok=True)
            self._output_file: TextIO = open(  # noqa: SIM115 # pylint: disable=consider-using-with
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
        self._data: dict[str, deque[float] | list[float]] = {}
        self._headers_windows: dict[str, int | None] = {}
        self._headers_minmax: dict[str, bool] = {}
        self._headers_delta: dict[str, bool] = {}
        self._current_row: dict[str, float] = {}

        if config is not None:
            self.save_config(config)
            self._config: Config = config

        self._use_tensorboard: bool = use_tensorboard
        self._use_wandb: bool = use_wandb

        if self._use_tensorboard and self._maste_proc:
            self._tensorboard_writer = SummaryWriter(log_dir=os.path.join(self._log_dir, 'tb'))

        if self._use_wandb and self._maste_proc:  # pragma: no cover
            project: str = self._config.logger_cfgs.get('wandb_project', 'omnisafe')
            name: str = f'{exp_name}-{relpath}'
            print('project', project, 'name', name)
            wandb.init(
                project=project,
                name=name,
                dir=self._log_dir,
                config=config,
            )
            if config is not None:
                wandb.config.update(config)  # type: ignore
            if models is not None:
                for model in models:
                    wandb.watch(model)  # type: ignore

    def log(self, msg: str, color: str = 'green', bold: bool = False) -> None:
        """Log the message to the console and the file.

        Args:
            msg (str): The message to be logged.
            color (str, optional): The color of the message. Defaults to 'green'.
            bold (bool, optional): Whether the message is bold. Defaults to False.
        """
        if self._maste_proc:
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
            what_to_save (dict[str, Any]): The dict of the things to be saved.
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

        .. code-block:: text

            ----------------------------------------------------
            |       Name            |            Value         |
            ----------------------------------------------------
            |    Train/Epoch        |             25           |
            |  Metrics/EpCost/Min   |            22.38         |
            |  Metrics/EpCost/Max   |            25.48         |
            |  Metrics/EpCost/Mean  |            23.93         |
            ----------------------------------------------------

        Args:
            key (str): The name of the key.
            window_length (int or None, optional): The length of the window. Defaults to None.
            min_and_max (bool, optional): Whether to record the min and max value. Defaults to False.
            delta (bool, optional): Whether to record the delta value. Defaults to False.
        """
        assert key not in self._current_row, f'Key {key} has been registered'
        self._current_row[key] = 0
        if min_and_max:
            self._current_row[f'{key}/Min'] = 0
            self._current_row[f'{key}/Max'] = 0
            self._current_row[f'{key}/Std'] = 0
            self._headers_minmax[key] = True
            self._headers_minmax[f'{key}/Min'] = False
            self._headers_minmax[f'{key}/Max'] = False
            self._headers_minmax[f'{key}/Std'] = False

        else:
            self._headers_minmax[key] = False

        if delta:
            self._current_row[f'{key}/Delta'] = 0
            self._headers_delta[key] = True
            self._headers_delta[f'{key}/Delta'] = False
            self._headers_minmax[f'{key}/Delta'] = False
        else:
            self._headers_delta[key] = False

        if window_length is not None:
            self._data[key] = deque(maxlen=window_length)
            self._headers_windows[key] = window_length
        else:
            self._data[key] = []
            self._headers_windows[key] = None

    def store(
        self,
        data: dict[str, Any] | None = None,
        /,
        **kwargs: Any | float | np.ndarray | torch.Tensor,
    ) -> None:
        """Store the data to the logger.

        .. note ::
            The data stored in ``data`` will be updated by ``kwargs``.

        Args:
            data (dict[str, float | np.ndarray | torch.Tensor] or None, optional): The data to
                be stored. Defaults to None.

        Keyword Args:
            kwargs (int, float, np.ndarray, or torch.Tensor): The data to be stored.
        """
        if data is not None:
            kwargs.update(data)
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

        - If the key is registered with window_length, the data will be averaged in the window.
        - Write the data to the csv file.
        - Write the data to the tensorboard.
        - Update the progress logger.
        """
        self._update_current_row()
        table = Table('Metrics', 'Value')
        if self._maste_proc:
            self._epoch += 1
            key_lens = list(map(len, self._current_row.keys()))
            max_key_len = max(15, *key_lens)
            for key, val in self._current_row.items():
                if self._headers_minmax[key]:
                    table.add_row(f'{key}/Mean'[:max_key_len], str(val)[:max_key_len])
                else:
                    table.add_row(key[:max_key_len], str(val)[:max_key_len])

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
            old_data = self._current_row[key]
            if self._headers_minmax[key]:
                mean, min_val, max_val, std = self.get_stats(key, True)
                self._current_row[key] = mean
                self._current_row[f'{key}/Min'] = min_val
                self._current_row[f'{key}/Max'] = max_val
                self._current_row[f'{key}/Std'] = std
            else:
                mean = self.get_stats(key, False)[0]
                self._current_row[key] = mean

            if self._headers_delta[key]:
                self._current_row[f'{key}/Delta'] = mean - old_data

            if self._headers_windows[key] is None:
                self._data[key] = []

    def get_stats(
        self,
        key: str,
        min_and_max: bool = False,
    ) -> tuple[float, ...]:
        """Get the statistics of the key.

        Args:
            key (str): The key to be registered.
            min_and_max (bool, optional): Whether to record the min and max value of the key.
                Defaults to False.

        Returns:
            The mean value of the key or (mean, min, max, std) of the key.
        """
        assert key in self._current_row, f'Key {key} has not been registered'
        vals = self._data[key]
        if isinstance(vals, deque):
            vals = list(vals)

        if min_and_max:
            mean, std, min_val, max_val = dist_statistics_scalar(
                torch.tensor(vals).to(os.getenv('OMNISAFE_DEVICE', 'cpu')),
                with_min_and_max=True,
            )
            return mean.item(), min_val.mean().item(), max_val.mean().item(), std.item()

        mean, std = dist_statistics_scalar(  # pylint: disable=unbalanced-tuple-unpacking
            torch.tensor(vals).to(os.getenv('OMNISAFE_DEVICE', 'cpu')),
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
