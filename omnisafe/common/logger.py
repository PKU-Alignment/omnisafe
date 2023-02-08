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
import json
import os
import os.path as osp
import time
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from omnisafe.utils.distributed_utils import mpi_statistics_scalar, proc_id
from omnisafe.utils.logger_utils import colorize, convert_json


# pylint: disable-next=too-many-instance-attributes
class Logger:
    """Implementation of the Logger.

    A logger to record the training process.
    It can record the training process to a file and print it to the console.
    It can also record the training process to tensorboard.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        exp_name: str,
        output_fname: str = 'progress.txt',
        debug: bool = False,
        level: int = 1,
        datestamp: int = True,
        hms_time: int = time.strftime('%Y-%m-%d_%H-%M-%S'),
        use_tensor_board: bool = True,
        verbose: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize the logger.

        Args:
            data_dir (str): The directory to save the config and progress.
            exp_name (str): The name of the experiment.
            output_fname (str, optional): The name of the progress file. Defaults to 'progress.txt'.
            debug (bool, optional): Whether to print debug information. Defaults to False.
            level (int, optional): The level of the logger. Defaults to 1.
            datestamp (int, optional): Whether to add datestamp to the log directory. Defaults to True.
            hms_time (int, optional): The time of the experiment. Defaults to time.strftime('%Y-%m-%d_%H-%M-%S').
            use_tensor_board (bool, optional): Whether to use tensorboard. Defaults to True.
            verbose (bool, optional): Whether to print the progress to the console. Defaults to True.
            seed (int, optional): The seed of the experiment. Defaults to 0.
        """
        relpath = hms_time if datestamp else ''
        if seed is not None:
            subfolder = '-'.join(['seed', str(seed).zfill(3)])
            relpath = '-'.join([subfolder, relpath])
        self.log_dir = os.path.join(data_dir, exp_name, relpath)
        self.debug = debug if proc_id() == 0 else False
        self.level = level
        # only the MPI root process is allowed to print information to console
        self.verbose = verbose if proc_id() == 0 else False

        if proc_id() == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            # pylint: disable-next=consider-using-with
            self.output_file = open(
                osp.join(self.log_dir, output_fname), encoding='utf-8', mode='w'
            )
            atexit.register(self.output_file.close)
            print(colorize(f'Logging data to {self.output_file.name}', 'cyan', bold=True))
        else:
            self.output_file = None

        self.epoch = 0
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.torch_saver_elements = None

        # Setup tensor board logging if enabled and MPI root process
        self.summary_writer = (
            SummaryWriter(os.path.join(self.log_dir, 'tb'))
            if use_tensor_board and proc_id() == 0
            else None
        )

        self.epoch_dict = {}

    def log(
        self,
        msg: str,
        color: str = 'green',
    ) -> None:
        """Print a colorized message to stdout."""
        if self.verbose and self.level > 0:
            print(colorize(msg, color, bold=False))

    def store(self, **kwargs) -> None:
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical values.
        """
        for key, value in kwargs.items():
            if key not in self.epoch_dict:
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(value)

    def log_single_value(self, key: str, val: str) -> None:
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using :meth:`log_tabular` to store values for each diagnostic,
        make sure to call :meth:`dump_tabular` to write them out to file,
        and stdout (otherwise they will not get saved anywhere).

        Args:
            key (str): The name of the diagnostic.
            val (str): The value of the diagnostic.
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert (
                key in self.log_headers
            ), f"Trying to introduce a new key {key} that you didn't include in the first iteration"
        assert (
            key not in self.log_current_row
        ), f'You already set {key} this iteration. Maybe you forgot to call dump_tabular()'
        self.log_current_row[key] = val

    def log_tabular(
        self, key: str, val: str = None, min_and_max: bool = False, std: bool = False
    ) -> None:
        """Log a value of some diagnostic.

        Args:
            key (str): The name of the diagnostic.
            val (str, optional): The value of the diagnostic. Defaults to None.
            min_and_max (bool, optional): Whether to log the min and max values. Defaults to False.
            std (bool, optional): Whether to log the standard deviation. Defaults to False.
        """

        if val is not None:
            self.log_single_value(key, val)
        else:
            stats = self.get_stats(key, min_and_max)
            if min_and_max or std:
                self.log_single_value(key + '/Mean', stats[0].numpy())
            else:
                self.log_single_value(key, stats[0].numpy())
            if std:
                self.log_single_value(key + '/Std', stats[1].numpy())
            if min_and_max:
                self.log_single_value(key + '/Min', stats[2].numpy())
                self.log_single_value(key + '/Max', stats[3].numpy())
        self.epoch_dict[key] = []

    def save_config(self, config) -> None:
        """Save the configuration of the experiment."""

        if proc_id() == 0:  # only root process logs configurations
            config_json = convert_json(config)
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            # if self.verbose and self.level > 0:
            #     print(colorize('Run with config:', color='yellow', bold=True))
            #     print(output)
            print(colorize('Save with config in config.json', color='yellow', bold=True))
            with open(osp.join(self.log_dir, 'config.json'), encoding='utf-8', mode='w') as out:
                out.write(output)

    def get_stats(
        self,
        key: str,
        with_min_and_max: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Get the statistics of a key.

        Args:
            key (str): The name of the diagnostic.
            with_min_and_max (bool, optional): Whether to log the min and max values. Defaults to False.
        """
        assert key in self.epoch_dict, f'key={key} not in dict'

        val = self.epoch_dict[key]
        vals = (
            np.concatenate(val) if isinstance(val[0], np.ndarray) and len(val[0].shape) > 0 else val
        )
        return mpi_statistics_scalar(torch.tensor(vals), with_min_and_max=with_min_and_max)

    def dump_tabular(self) -> None:
        """Write all of the diagnostics from the current iteration,
        both to stdout, and to the output file.

        If you want to add more information to the diagnostics,
        you should call :meth:`log_tabular` again after this.

        .. warning::
            This function should only be called once per iteration after :meth:`log_tabular`.
        """
        if proc_id() == 0:
            vals = []
            self.epoch += 1
            # Print formatted information into console
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))  # pylint: disable=nested-min-max
            keystr = '%' + '%d' % max_key_len  # pylint: disable=consider-using-f-string
            fmt = '| ' + keystr + 's | %15s |'
            n_slashes = 22 + max_key_len
            if self.verbose and self.level > 0:
                print('-' * n_slashes)
            # print('-' * n_slashes) if self.verbose and self.level > 0 else None
            for key in self.log_headers:
                val = self.log_current_row.get(key, '')
                # pylint: disable-next=consider-using-f-string
                valstr = '%8.3g' % val if hasattr(val, '__float__') else val
                if self.verbose and self.level > 0:
                    print(fmt % (key, valstr))
                vals.append(val)
            if self.verbose and self.level > 0:
                print('-' * n_slashes, flush=True)

            # Write into the output file (can be any text file format, e.g. CSV)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write(' '.join(self.log_headers) + '\n')
                self.output_file.write(' '.join(map(str, vals)) + '\n')
                self.output_file.flush()

            if self.summary_writer is not None:
                for key, value in zip(self.log_headers, vals):
                    self.summary_writer.add_scalar(key, value, global_step=self.epoch)

                # Flushes the event file to disk. Call this method to make sure
                # that all pending events have been written to disk.
                self.summary_writer.flush()

        # free logged information in all processes...
        self.log_current_row.clear()
        self.first_row = False

        # Check if all values from dict are dumped -> prevent memory overflow
        for key, value in self.epoch_dict.items():
            if len(value) > 0:
                print(f'epoch_dict: key={key} was not logged.')

    def setup_torch_saver(self, what_to_save: dict) -> None:
        """Setup the torch saver."""
        self.torch_saver_elements = what_to_save

    def torch_save(self, itr: str = None) -> None:
        """
        Saves the PyTorch model (or models).
        """
        if proc_id() == 0:
            assert (
                self.torch_saver_elements is not None
            ), 'First have to setup saving with self.setup_torch_saver'
            fpath = 'torch_save'
            fpath = osp.join(self.log_dir, fpath)
            fname = f'model {itr}.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)

            params = {
                k: v.state_dict() if isinstance(v, torch.nn.Module) else v
                for k, v in self.torch_saver_elements.items()
            }
            torch.save(params, fname)

    def close(self) -> None:
        """Close the logger.

        Close opened output files immediately after training in order to
        avoid number of open files overflow. Avoids the following error:
        ``OSError: [Errno 24] Too many open files``.
        """
        if proc_id() == 0:
            self.output_file.close()
