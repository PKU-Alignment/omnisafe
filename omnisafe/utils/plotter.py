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
"""Plotter class for plotting data from experiments."""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


class Plotter:
    """Plotter class for plotting data from experiments.

    Suppose you have run several experiments, with the aim of comparing performance between
    different algorithms, resulting in a log directory structure of:

    .. code-block:: text

        runs/
            SafetyAntVelocity-v1/
                CPO/
                    seed0/
                    seed5/
                    seed10/
                PCPO/
                    seed0/
                    seed5/
                    seed10/
            SafetyHalfCheetahVelocity-v1/
                CPO/
                    seed0/
                    seed5/
                    seed10/
                PCPO/
                    seed0/
                    seed5/
                    seed10/

    Examples:
        You can easily produce a graph comparing CPO and PCPO in 'SafetyAntVelocity-v1' with:

        .. code-block:: bash

            python plot.py './runs/SafetyAntVelocity-v1/'

    Attributes:
        div_line_width (int): The width of the dividing line between subplots.
        exp_idx (int): The index of the experiment.
        units (dict[str, Any]): The units of the data.
    """

    def __init__(self) -> None:
        """Initialize an instance of :class:`Plotter`."""
        self.div_line_width: int = 50
        self.exp_idx: int = 0
        self.units: dict[str, Any] = {}

    def plot_data(
        self,
        sub_figures: np.ndarray,
        data: list[DataFrame],
        xaxis: str = 'Steps',
        value: str = 'Rewards',
        condition: str = 'Condition1',
        smooth: int = 1,
        **kwargs: Any,
    ) -> None:
        """Plot data from a pandas dataframe.

        .. note::
            The ``smooth`` means smoothing the data with moving window average.

        Example:
            >>> smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])

        where the "smooth" param is width of that window (2k+1)

        Args:
            sub_figures (np.ndarray): The subplots.
            data (list of DataFrame): The data to plot.
            xaxis (str, optional): The x-axis label. Defaults to 'Steps'.
            value (str, optional): The y-axis label. Defaults to 'Rewards'.
            condition (str, optional): The condition label. Defaults to 'Condition1'.
            smooth (int, optional): The smoothing window size. Defaults to 1.

        Keyword Args:
            kwargs: Other keyword arguments for ``sns.lineplot``.
        """
        if smooth > 1:
            y = np.ones(smooth)
            for datum in data:
                x = np.asarray(datum[value])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                datum[value] = smoothed_x
                x = np.asarray(datum['Costs'])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                datum['Costs'] = smoothed_x

        data_to_plot = pd.concat(data, ignore_index=True)
        sns.lineplot(
            data=data_to_plot,
            x=xaxis,
            y='Rewards',
            hue=condition,
            errorbar='sd',
            ax=sub_figures[0],
            **kwargs,
        )
        sns.lineplot(
            data=data_to_plot,
            x=xaxis,
            y='Costs',
            hue=condition,
            errorbar='sd',
            ax=sub_figures[1],
            **kwargs,
        )
        # plt.legend(loc='best').set_draggable(True)
        # plt.legend(loc='upper center', ncol=3, handlelength=1,
        #           borderaxespad=0., prop={'size': 13})
        sub_figures[0].legend(
            loc='upper center',
            ncol=6,
            handlelength=1,
            mode='expand',
            borderaxespad=0.0,
            prop={'size': 13},
        )
        sub_figures[1].legend(
            loc='upper center',
            ncol=6,
            handlelength=1,
            mode='expand',
            borderaxespad=0.0,
            prop={'size': 13},
        )

        xscale = np.max(np.asarray(data_to_plot[xaxis], dtype=np.int32)) > 5e3
        if xscale:
            # just some formatting niceness: x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.tight_layout(pad=0.5)

    def get_datasets(self, logdir: str, condition: str | None = None) -> list[DataFrame]:
        """Recursively look through logdir for files named "progress.txt".

        Assumes that any file "progress.txt" is a valid hit.

        Args:
            logdir (str): The directory to search for progress.txt files
            condition (str or None, optional): The condition label. Defaults to None.

        Returns:
            The datasets.

        Raise:
            FileNotFoundError: If the config file is not found.
            FileNotFoundError: If could not read from ``progress.csv`` file.
            ValueError: If no Train/Epoch column in progress.csv.
        """
        datasets: list[DataFrame] = []
        for root, _, files in os.walk(logdir):
            if 'progress.csv' in files:
                exp_name = None
                steps_per_epoch = None
                try:
                    with open(os.path.join(root, 'config.json'), encoding='utf-8') as f:
                        config = json.load(f)
                        if 'exp_name' in config:
                            exp_name = config['algo']
                            steps_per_epoch = config['algo_cfgs']['steps_per_epoch']
                except FileNotFoundError as error:
                    config_path = os.path.join(root, 'config.json')
                    raise FileNotFoundError(f'Could not read from {config_path}') from error
                condition1 = condition or exp_name or 'exp'
                condition2 = condition1 + '-' + str(self.exp_idx)
                self.exp_idx += 1
                if condition1 not in self.units:
                    self.units[condition1] = 0
                unit = self.units[condition1]
                self.units[condition1] += 1
                try:
                    exp_data = pd.read_csv(os.path.join(root, 'progress.csv'))

                except FileNotFoundError as error:
                    progress_path = os.path.join(root, 'progress.csv')
                    raise FileNotFoundError(f'Could not read from {progress_path}') from error
                performance = (
                    'Metrics/TestEpRet' if 'Metrics/TestEpRet' in exp_data else 'Metrics/EpRet'
                )
                cost_performance = (
                    'Metrics/TestEpCost' if 'Metrics/TestEpCost' in exp_data else 'Metrics/EpCost'
                )
                exp_data.insert(len(exp_data.columns), 'Unit', unit)
                exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
                exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
                exp_data.insert(len(exp_data.columns), 'Rewards', exp_data[performance])
                exp_data.insert(len(exp_data.columns), 'Costs', exp_data[cost_performance])
                epoch = exp_data.get('Train/Epoch')
                if epoch is None or steps_per_epoch is None:
                    raise ValueError('No Train/Epoch column in progress.csv')
                exp_data.insert(
                    len(exp_data.columns),
                    'Steps',
                    epoch * steps_per_epoch,
                )
                datasets.append(exp_data)
        return datasets

    def get_all_datasets(
        self,
        all_logdirs: list[str],
        legend: list[str] | None = None,
        select: str | None = None,
        exclude: str | None = None,
    ) -> list[DataFrame]:
        """Get all the data from all the log directories.

        For every entry in all_logdirs.
            1) check if the entry is a real directory and if it is, pull data from it;
            2) if not, check to see if the entry is a prefix for a real directory, and pull data from that.

        Args:
            all_logdirs (list of str): List of log directories.
            legend (list of str or None, optional): List of legend names. Defaults to None.
            select (str or None, optional): Select logdirs that contain this string. Defaults to None.
            exclude (str or None, optional): Exclude logdirs that contain this string. Defaults to None.

        Returns:
            All the data stored in a list of DataFrames.
        """
        logdirs = []
        for logdir in all_logdirs:
            if osp.isdir(logdir) and logdir[-1] == os.sep:
                logdirs += [logdir]
            else:
                basedir = osp.dirname(logdir)
                prefix = logdir.split(os.sep)[-1]
                listdir = os.listdir(basedir)
                logdirs += sorted([osp.join(basedir, x) for x in listdir if prefix in x])

        # Enforce selection rules, which check logdirs for certain sub strings.
        # Makes it easier to look at graphs from particular ablations, if you
        # launch many jobs at once with similar names.
        if select is not None:
            logdirs = [log for log in logdirs if all(x in log for x in select)]
        if exclude is not None:
            logdirs = [log for log in logdirs if all(x not in log for x in exclude)]

        # verify logdirs
        print('Plotting from...\n' + '=' * self.div_line_width + '\n')
        for logdir in logdirs:
            print(logdir)
        print('\n' + '=' * self.div_line_width)

        # make sure the legend is compatible with the logdirs
        assert not (legend) or (
            len(legend) == len(logdirs)
        ), 'Must give a legend title for each set of experiments.'

        # load data from logdirs
        data = []
        if legend:
            for log, leg in zip(logdirs, legend):
                data += self.get_datasets(log, leg)
        else:
            for log in logdirs:
                data += self.get_datasets(log)
        return data

    # pylint: disable-next=too-many-arguments
    def make_plots(
        self,
        all_logdirs: list[str],
        legend: list[str] | None = None,
        xaxis: str = 'Steps',
        value: str = 'Rewards',
        count: bool = False,
        cost_limit: float | None = None,
        smooth: int = 1,
        select: str | None = None,
        exclude: str | None = None,
        estimator: str = 'mean',
        save_dir: str = './',
        save_name: str | None = None,
        save_format: str = 'png',
        show_image: bool = False,
    ) -> None:
        """Make plots from the data in the specified log directories.

        Args:
            all_logdirs (list of str): As many log directories (or prefixes to log directories,
                which the plotter will automatically complete internally) as you'd like to plot from.
            legend (list of str or None, optional): Optional way to specify legend for the plot. The
                plotter legend will automatically use the ``exp_name`` from the config.json file,
                unless you tell it otherwise through this flag. This only works if you provide a
                name for each directory that will get plotted. (Note: this may not be the same as
                the number of logdir args you provide! Recall that the plotter looks for
                autocompletes of the logdir args: there may be more than one match for a given
                logdir prefix, and you will need to provide a legend string for each one of those
                matches--unless you have removed some of them as candidates via selection or
                exclusion rules (below).)
            xaxis (str, optional): Pick what column from data is used for the x-axis. Defaults to
                ``TotalEnvInteracts``.
            value (str, optional): Pick what columns from data to graph on the y-axis. Submitting
                multiple values will produce multiple graphs. Defaults to ``Performance``, which is
                not an actual output of any algorithm. Instead, ``Performance`` refers to either
                ``AverageEpRet``, the correct performance measure for the on-policy algorithms, or
                ``AverageTestEpRet``, the correct performance measure for the off-policy algorithms.
                The plotter will automatically figure out which of ``AverageEpRet`` or
                ``AverageTestEpRet`` to report for each separate logdir.
            count (bool, optional): Optional flag. By default, the plotter shows y-values which are
                averaged across all results that share an ``exp_name``, which is typically a set of
                identical experiments that only vary in random seed. But if you'd like to see all of
                those curves separately, use the ``--count`` flag.
            cost_limit (float or None, optional): Optional way to specify the cost limit of the plot.
                Defaults to ``None``, which means the plot will not have a cost limit.
            smooth (int, optional): Smooth data by averaging it over a fixed window. This parameter
                says how wide the averaging window will be.
            select (str or None, optional): Optional selection rule: the plotter will only show
                curves from logdirs that contain all of these sub strings.
            exclude (str or None, optional): Optional exclusion rule: plotter will only show curves
                from logdirs that do not contain these sub strings.
            estimator (str, optional): Optional way to specify how to aggregate data across multiple
                runs. Defaults to ``mean``.
            save_dir (str, optional): Optional way to specify where to save the plot. Defaults to
                ``./``.
            save_name (str or None, optional): Optional way to specify the name of the plot.
                Defaults to ``None``, which means the plot will be saved with the name of the first
                logdir.
            save_format (str, optional): Optional way to specify the format of the plot. Defaults
                to ``png``.
            show_image (bool, optional): Optional flag. If set, the plot will be displayed on screen.
                Defaults to ``False``.
        """
        assert xaxis is not None, 'Must specify xaxis'
        data = self.get_all_datasets(all_logdirs, legend, select, exclude)
        condition = 'Condition2' if count else 'Condition1'
        # choose what to show on main curve: mean? max? min?
        estimator = getattr(np, estimator)
        sns.set()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 5),
        )
        self.plot_data(
            axes,
            data,
            xaxis=xaxis,
            value=value,
            condition=condition,
            smooth=smooth,
            estimator=estimator,
        )
        if cost_limit:
            axes[1].axhline(y=cost_limit, ls='--', c='black', linewidth=2)
        if save_name is None:
            save_name = all_logdirs[0].split('/')[-1]
        if show_image:
            plt.show()
        fig.savefig(
            os.path.join(save_dir, f'{save_name}.{save_format}'),
            bbox_inches='tight',
            pad_inches=0.0,
        )
