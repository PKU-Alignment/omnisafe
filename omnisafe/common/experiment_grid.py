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
"""Implementation of the Experiment Grid."""

from __future__ import annotations

import json
import os
import string
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from textwrap import dedent
from typing import Any

import numpy as np
from rich.console import Console
from tqdm import trange

from omnisafe.algorithms import ALGORITHM2TYPE
from omnisafe.common.statistics_tools import StatisticsTools
from omnisafe.evaluator import Evaluator
from omnisafe.utils.exp_grid_tools import all_bools, valid_str
from omnisafe.utils.tools import (
    assert_with_exit,
    hash_string,
    load_yaml,
    recursive_check_config,
    recursive_dict2json,
)


# pylint: disable-next=too-many-instance-attributes
class ExperimentGrid:
    """Tool for running many experiments given hyper-parameters ranges."""

    def __init__(self, exp_name='') -> None:
        """Initialize the ExperimentGrid.

        Args:
            exp_name (str): Name of the experiment grid.
        """
        self.keys: list[str] = []
        self.vals: list[Any] = []
        self.shs: list[str] = []
        self.in_names: list[bool] = []
        self.div_line_width = 80
        assert isinstance(exp_name, str), 'Name has to be a string.'
        self.name = exp_name
        self._console = Console()
        self.log_dir: str

        # Whether GridSearch provides automatically-generated default shorthands
        self.default_shorthand = True

        # Tells the GridSearch how many seconds to pause for before launching experiments
        self.wait_defore_launch = 0

        # Whether to automatically insert a date and time stamp into the names of save directories
        self.foce_datastamp = False

        self._statistical_tools: StatisticsTools
        self._evaluator: Evaluator

    def print(self) -> None:
        """Print a helpful report about the experiment grid.

        This function prints a helpful report about the experiment grid, including
        the name of the experiment grid, the parameters being varied, and the
        possible values for each parameter.

        In Command Line:

        .. code-block:: bash

            ===================== ExperimentGrid [test] runs over parameters: =====================
            env_name                                [env]
            ['SafetyPointGoal1-v0', 'MountainCar-v0', 'Acrobot-v1']
            algo                               [algo]
            ['SAC', 'DDPG', 'TD3']
            seed                                    [seed]
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        print('=' * self.div_line_width)

        # Prepare announcement at top of printing. If the ExperimentGrid has a
        # short name, write this as one line. If the name is long, break the
        # announcement over two lines.
        base_msg = 'ExperimentGrid %s runs over parameters:\n'
        name_insert = '[' + self.name + ']'
        if len(base_msg % name_insert) <= 80:
            msg = base_msg % name_insert
        else:
            msg = base_msg % (name_insert + '\n')
        self._console.print(msg, style='green bold')

        # List off parameters, shorthands, and possible values.
        for key, value, shorthand in zip(self.keys, self.vals, self.shs):
            self._console.print('', key.ljust(40), style='cyan bold', end='')
            print('[' + shorthand + ']' if shorthand is not None else '', '\n')
            for _, val in enumerate(value):
                print('\t' + json.dumps(val, indent=4, sort_keys=True))
            print()

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_total = int(np.prod([len(v) for v in self.vals]))
        if 'seed' in self.keys:
            num_seeds = len(self.vals[self.keys.index('seed')])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(' Variants, counting seeds: '.ljust(40), nvars_total)
        print(' Variants, not counting seeds: '.ljust(40), nvars_seedless)
        print()
        print('=' * self.div_line_width)

    def _default_shorthand(self, key):
        r"""Default shorthand.
        Create a default shorthand for the key, built from the first
        three letters of each colon-separated part.
        But if the first three letters contains something which isn't
        alphanumeric, shear that off.

        Example:
            >>> _default_shorthand('env_name:SafetyPointGoal1-v0')
            'env'

        Args:
            key (string): Name of parameter.
        """

        valid_chars = f'{string.ascii_letters}{string.digits}'

        def shear(value):
            return ''.join(z for z in value[:3] if z in valid_chars)

        return '-'.join([shear(x) for x in key.split(':')])

    def add(self, key, vals, shorthand: str | None = None, in_name: bool = False):
        r"""Add a parameter (key) to the grid config, with potential values (vals).

        By default, if a shorthand isn't given, one is automatically generated
        from the key using the first three letters of each colon-separated
        term.

        .. hint::

            This function is called in ``omnisafe/examples/benchmarks/run_experiment_grid.py``.

        Example:
            >>> add('env_id', ['SafetyPointGoal1-v0', 'MountainCar-v0', 'Acrobot-v1'])
            >>> add('algo', ['SAC', 'DDPG', 'TD3'])
            >>> add('seed', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        Args:
            key (string): Name of parameter.

            vals (value or list of values): Allowed values of parameter.

            shorthand (string): Optional, shortened name of parameter. For
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``.

            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), 'Key must be a string.'
        if not isinstance(vals, list):
            vals = [vals]
        if self.default_shorthand and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        assert len(set(self.keys)) == len(self.keys), f'Duplicate key: `{key}`'
        assert len(set(vals)) == len(vals), f'Duplicate values in {vals} for key: `{key}`'
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        r"""Given a variant (dict of valid param/value pairs), make an exp_name.

        A variant's name is constructed as the grid name (if you've given it
        one), plus param names (or shorthands if available) and values
        separated by underscores.

        ..warning::
            if ``seed`` is a parameter, it is not included in the name.

        Example:
            >>> variant_name({'env_id': 'SafetyPointGoal1-v0', 'algo': 'SAC', 'seed': 0})
            exp_name = 'SafetyPointGoal1-v0_SAC_0'

        Args:
            variant (dict): Variant dictionary.
        """

        def get_val(value, key):
            # utility method for getting the correct value out of a variant
            # given as a nested dict. Assumes that a parameter name, k,
            # describes a path into the nested dict, such that k='a:b:c'
            # corresponds to value=variant['a']['b']['c']. Uses recursion
            # to get this
            print('value', value, 'key', key)
            if key in value:
                return value[key]

            splits = key.split(':')
            k_0, k_1 = splits[0], ':'.join(splits[1:])
            return get_val(value[k_0], k_1)

        # start the name off with the name of the variant generator.
        var_name = self.name

        # build the rest of the name by looping through all parameters,
        # and deciding which ones need to go in there.
        for key, value, shorthand, inn in zip(self.keys, self.vals, self.shs, self.in_names):
            # Include a parameter in a name if either 1) it can take multiple
            # values, or 2) the user specified that it must appear in the name.
            # Except, however, when the parameter is 'seed'. Seed is handled
            # differently so that runs of the same experiment, with different
            # seeds, will be grouped by experiment name.
            if (len(value) > 1 or inn) and key != 'seed':
                # use the shorthand if available, otherwise the full name.
                param_name = shorthand if shorthand is not None else key
                param_name = valid_str(param_name)
                # Get variant value for parameter k
                variant_val = get_val(variant, key)

                # append to name
                if all_bools(value):
                    # if this is a param which only takes boolean values,
                    # only include in the name if it's True for this variant.
                    var_name += ('_' + param_name) if variant_val else ''
                else:
                    var_name += '_' + param_name + valid_str(variant_val)

        return var_name.lstrip('_')

    def update_dict(self, total_dict, item_dict):
        """Updater of multi-level dictionary.

        This function is used to update the total dictionary with the item
        dictionary.

        Args:
            total_dict (dict): Total dictionary.
            item_dict (dict): Item dictionary.
        """
        for idd in item_dict:
            total_value = total_dict.get(idd)
            item_value = item_dict.get(idd)

            if total_value is None:
                total_dict.update({idd: item_value})
            elif isinstance(item_value, dict):
                self.update_dict(total_value, item_value)
                total_dict.update({idd: total_value})
            else:
                total_value = item_value
                total_dict.update({idd: total_value})

    def _variants(self, keys, vals):
        """Recursively builds list of valid variants.

        Args:
            keys (list): List of keys.
            vals (list): List of values.
        """
        if len(keys) == 1:
            pre_variants: list[dict] = [{}]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                current_variants = deepcopy(pre_v)
                v_temp = {}
                key_list = keys[0].split(':')
                v_temp[key_list[-1]] = val
                for key in reversed(key_list[:-1]):
                    v_temp = {key: v_temp}
                self.update_dict(current_variants, v_temp)
                variants.append(current_variants)

        return variants

    def variants(self):
        r"""Makes a list of dict, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take
        the form ``'full:param:name'``.

        The colons are taken to indicate that these parameters should
        have a nested dict structure. eg, if there are two params,

        .. hint::

            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:a'``    1
            ``'base:param:b'``    2
            ``'base:param:c'``    3
            ``'special:d'``       4
            ``'special:e'``       5
            ====================  ===

        the variant dict will have the structure

        .. parsed-literal::

            {
                'base': {
                    'param': {
                        'a': 1,
                        'b': 2,
                        'c': 3
                    }
                },
                'special': {
                    'd': 4,
                    'e': 5
                }
            }
        """
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            """Build the full nested dict version of var, based on key names."""
            new_var: dict = {}
            unflatten_set = set()

            for key, value in var.items():
                if ':' in key:
                    splits = key.split(':')
                    k_0 = splits[0]
                    assert k_0 not in new_var or isinstance(
                        new_var[k_0],
                        dict,
                    ), "You can't assign multiple values to the same key."

                    if k_0 not in new_var:
                        new_var[k_0] = {}

                    sub_k = ':'.join(splits[1:])
                    new_var[k_0][sub_k] = value
                    unflatten_set.add(k_0)
                else:
                    assert key not in new_var, "You can't assign multiple values to the same key."
                    new_var[key] = value

            # make sure to fill out the nested dict.
            for key in unflatten_set:
                new_var[key] = unflatten_var(new_var[key])

            return new_var

        return [unflatten_var(var) for var in flat_variants]

    # pylint: disable-next=too-many-locals
    def run(self, thunk, num_pool=1, parent_dir=None, is_test=False, gpu_id=None):
        r"""Run each variant in the grid with function 'thunk'.

        Note: 'thunk' must be either a callable function, or a string. If it is
        a string, it must be the name of a parameter whose values are all
        callable functions.

        Uses ``call_experiment`` to actually launch each experiment, and gives
        each variant a name using ``self.variant_name()``.

        Maintenance note: the args for ExperimentGrid.run should track closely
        to the args for call_experiment. However, ``seed`` is omitted because
        we presume the user may add it as a parameter in the grid.
        """
        if parent_dir is None:
            self.log_dir = os.path.join('./', 'exp-x', self.name)
        else:
            self.log_dir = os.path.join(parent_dir, self.name)
        assert_with_exit(
            not os.path.exists(self.log_dir),
            (
                f'log_dir {self.log_dir} already exists!'
                'please make sure that you are not overwriting an existing experiment,'
                'it is important for analyzing the results of the experiment.'
            ),
        )
        self.save_grid_config()
        # print info about self.
        self.print()

        # make the list of all variants.
        variants = self.variants()

        # print variant names for the user.
        var_names = {self.variant_name(var) for var in variants}
        var_names = sorted(var_names)
        line = '=' * self.div_line_width

        self._console.print('\nPreparing to run the following experiments...', style='bold green')
        joined_var_names = '\n'.join(var_names)
        announcement = f'\n{joined_var_names}\n\n{line}'
        print(announcement)

        if self.wait_defore_launch > 0:
            self._console.print(
                dedent(
                    """
                    Launch delayed to give you a few seconds to review your experiments.

                    To customize or disable this behavior, change WAIT_BEFORE_LAUNCH in
                    spinup/user_config.py.

                    """,
                ),
                style='cyan, bold',
                end='',
            )
            print(line)
            wait, steps = self.wait_defore_launch, 100
            prog_bar = trange(
                steps,
                desc='Launching in...',
                leave=False,
                ncols=self.div_line_width,
                mininterval=0.25,
                bar_format='{desc}: {bar}| {remaining} {elapsed}',
            )
            for _ in prog_bar:
                time.sleep(wait / steps)

        pool = Pool(max_workers=num_pool)
        # run the variants.
        results = []
        exp_names = []

        for idx, var in enumerate(variants):
            self.check_variant_vaild(var)
            print('current_config', var)
            clean_var = deepcopy(var)
            clean_var.pop('seed', None)
            if gpu_id is not None:
                device_id = gpu_id[idx % len(gpu_id)]
                device = f'cuda:{device_id}'
                self.update_dict(var, {'train_cfgs': {'device': device}})
            exp_name = recursive_dict2json(clean_var)
            hashed_exp_name = var['env_id'][:30] + '---' + hash_string(exp_name)
            exp_names.append(':'.join((hashed_exp_name[:5], exp_name)))
            exp_log_dir = os.path.join(self.log_dir, hashed_exp_name, '')
            var['logger_cfgs'].update({'log_dir': exp_log_dir})
            self.save_same_exps_config(exp_log_dir, var)
            results.append(pool.submit(thunk, idx, var['algo'], var['env_id'], var))
        pool.shutdown()

        if not is_test:
            self.save_results(exp_names, variants, results)
        self._init_statistical_tools()

    def save_results(self, exp_names, variants, results):
        """Save results to a file."""
        path = os.path.join(self.log_dir, 'exp-x-results.txt')
        str_len = max(len(exp_name) for exp_name in exp_names)
        exp_names = [exp_name.ljust(str_len) for exp_name in exp_names]
        with open(path, 'a+', encoding='utf-8') as f:
            for idx, _ in enumerate(variants):
                f.write(exp_names[idx] + ': ')
                reward, cost, ep_len = results[idx].result()
                f.write('reward:' + str(round(reward, 2)) + ',')
                f.write('cost:' + str(round(cost, 2)) + ',')
                f.write('ep_len:' + str(ep_len))
                f.write('\n')

    def save_same_exps_config(self, exps_log_dir, variant):
        """Save experiment grid configurations as json."""
        os.makedirs(exps_log_dir, exist_ok=True)
        path = os.path.join(exps_log_dir, 'exps_config.json')
        json_config = json.dumps(variant, indent=4)
        with open(path, encoding='utf-8', mode='a+') as f:
            f.write('\n' + json_config)

    def save_grid_config(self):
        """Save experiment grid configurations as json."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'grid_config.json')
        self._console.print(
            'Save with config of experiment grid in grid_config.json',
            style='yellow bold',
        )
        json_config = json.dumps(dict(zip(self.keys, self.vals)), indent=4)
        with open(path, encoding='utf-8', mode='w') as f:
            f.write(json_config)

    def check_variant_vaild(self, variant):
        """Check if the variant is valid."""
        path = os.path.dirname(os.path.abspath(__file__))
        algo_type = ALGORITHM2TYPE.get(variant['algo'], '')
        cfg_path = os.path.join(path, '..', 'configs', algo_type, f"{variant['algo']}.yaml")
        default_config = load_yaml(cfg_path)['defaults']
        recursive_check_config(variant, default_config, exclude_keys=('algo', 'env_id'))

    def _init_statistical_tools(self):
        """Initialize statistical tools."""
        self._statistical_tools = StatisticsTools()
        self._evaluator = Evaluator()

    def analyze(
        self,
        parameter: str,
        values: list = None,
        compare_num: int = None,
        cost_limit: float = None,
    ):
        """Analyze the experiment results.

        Args:
            parameter (str): name of parameter to analyze.
            values (list): specific values of attribute,
                if it is specified, will only compare values in it.
            compare_num (int): number of values to compare,
                if it is specified, will combine any potential combination to compare.
            cost_limit (float): value for one line showed on graph to indicate cost.

        .. Note::
            `values` and `compare_num` cannot be set at the same time.
        """
        assert self._statistical_tools is not None, 'Please run run() first!'
        self._statistical_tools.load_source(self.log_dir)
        self._statistical_tools.draw_graph(parameter, values, compare_num, cost_limit)

    def evaluate(self, num_episodes: int = 10, cost_criteria: float = 1.0):
        """Agent Evaluation.

        Args:
            num_episodes (int): number of episodes to evaluate.
            cost_criteria (float): cost criteria for evaluation.
        """
        assert self._evaluator is not None, 'Please run run() first!'
        # pylint: disable-next=too-many-nested-blocks
        for set_of_params in os.scandir(self.log_dir):
            if set_of_params.is_dir():
                for single_exp in os.scandir(set_of_params):
                    if single_exp.is_dir():
                        for single_seed in os.scandir(single_exp):
                            for model in os.scandir(os.path.join(single_seed, 'torch_save')):
                                if model.is_file() and model.name.split('.')[-1] == 'pt':
                                    self._evaluator.load_saved(
                                        save_dir=single_seed,
                                        model_name=model.name,
                                    )
                                    self._evaluator.evaluate(
                                        num_episodes=num_episodes,
                                        cost_criteria=cost_criteria,
                                    )

    # pylint: disable-next=too-many-arguments
    def render(
        self,
        num_episodes: int = 10,
        render_mode: str = 'rgb_array',
        camera_name: str = 'track',
        width: int = 256,
        height: int = 256,
    ):
        """Evaluate and render some episodes.

        Args:
            num_episodes (int): number of episodes to render.
            render_mode (str): render mode, can be 'rgb_array', 'depth_array' or 'human'.
            camera_name (str): camera name, specify the camera which you use to capture
                images.
            width (int): width of the rendered image.
            height (int): height of the rendered image.
        """
        assert self._evaluator is not None, 'Please run run() first!'
        # pylint: disable-next=too-many-nested-blocks
        for set_of_params in os.scandir(self.log_dir):
            if set_of_params.is_dir():
                for single_exp in os.scandir(set_of_params):
                    if single_exp.is_dir():
                        for single_seed in os.scandir(single_exp):
                            for model in os.scandir(os.path.join(single_seed, 'torch_save')):
                                if model.is_file() and model.name.split('.')[-1] == 'pt':
                                    self._evaluator.load_saved(
                                        save_dir=single_seed,
                                        model_name=model.name,
                                        render_mode=render_mode,
                                        camera_name=camera_name,
                                        width=width,
                                        height=height,
                                    )
                                    self._evaluator.render(num_episodes=num_episodes)
