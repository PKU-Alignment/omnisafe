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
"""experiment grid"""

import os, sys
import time
import string
import numpy as np
import torch
import omnisafe
from textwrap import dedent
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor as Pool
from omnisafe.utils.logger_utils import colorize, convert_json
from omnisafe.utils.exp_grid_tools import all_bools, valid_str


class ExperimentGrid:
    """
    Tool for running many experiments given hyperparameter ranges.
    """

    def __init__(self, exp_name=''):
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.div_line_width = 80
        assert isinstance(exp_name, str), "Name has to be a string."
        self.name = exp_name

        # Whether GridSearch provides automatically-generated default shorthands:
        self.DEFAULT_SHORTHAND = True

        # Tells the GridSearch how many seconds to pause for before launching
        # experiments.
        self.WAIT_BEFORE_LAUNCH = 0

        # Where experiment outputs are saved by default:
        self.DEFAULT_DATA_DIR = os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data'
        )

        # Whether to automatically insert a date and time stamp into the names of
        # save directories:
        self.FORCE_DATESTAMP = False

    def print(self):
        """Print a helpful report about the experiment grid."""
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
        print(colorize(msg, color='green', bold=True))

        # List off parameters, shorthands, and possible values.
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = colorize(k.ljust(40), color='cyan', bold=True)
            print('', color_k, '[' + sh + ']' if sh is not None else '', '\n')
            for i, val in enumerate(v):
                print('\t' + str(convert_json(val)))
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
        # Create a default shorthand for the key, built from the first
        # three letters of each colon-separated part.
        # But if the first three letters contains something which isn't
        # alphanumeric, shear that off.
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)

        def shear(x):
            return ''.join(z for z in x[:3] if z in valid_chars)

        sh = '-'.join([shear(x) for x in key.split(':')])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        """
        Add a parameter (key) to the grid config, with potential values (vals).

        By default, if a shorthand isn't given, one is automatically generated
        from the key using the first three letters of each colon-separated
        term. To disable this behavior, change ``DEFAULT_SHORTHAND`` in the
        ``spinup/user_config.py`` file to ``False``.

        Args:
            key (string): Name of parameter.

            vals (value or list of values): Allowed values of parameter.

            shorthand (string): Optional, shortened name of parameter. For
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``.

            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(shorthand, str), "Shorthand must be a string."
        if not isinstance(vals, list):
            vals = [vals]
        if self.DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        """
        Given a variant (dict of valid param/value pairs), make an exp_name.

        A variant's name is constructed as the grid name (if you've given it
        one), plus param names (or shorthands if available) and values
        separated by underscores.

        Note: if ``seed`` is a parameter, it is not included in the name.
        """

        def get_val(v, k):
            # Utility method for getting the correct value out of a variant
            # given as a nested dict. Assumes that a parameter name, k,
            # describes a path into the nested dict, such that k='a:b:c'
            # corresponds to value=variant['a']['b']['c']. Uses recursion
            # to get this.
            if k in v:
                return v[k]
            else:
                splits = k.split(':')
                k0, k1 = splits[0], ':'.join(splits[1:])
                return get_val(v[k0], k1)

        # Start the name off with the name of the variant generator.
        var_name = self.name

        # Build the rest of the name by looping through all parameters,
        # and deciding which ones need to go in there.
        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):

            # Include a parameter in a name if either 1) it can take multiple
            # values, or 2) the user specified that it must appear in the name.
            # Except, however, when the parameter is 'seed'. Seed is handled
            # differently so that runs of the same experiment, with different
            # seeds, will be grouped by experiment name.
            if (len(v) > 1 or inn) and not (k == 'seed'):

                # Use the shorthand if available, otherwise the full name.
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)

                # Get variant value for parameter k
                variant_val = get_val(variant, k)

                # Append to name
                if all_bools(v):
                    # If this is a param which only takes boolean values,
                    # only include in the name if it's True for this variant.
                    var_name += ('_' + param_name) if variant_val else ''
                else:
                    var_name += '_' + param_name + valid_str(variant_val)

        return var_name.lstrip('_')

    def _variants(self, keys, vals):
        """
        Recursively builds list of valid variants.
        """
        if len(keys) == 1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = {}
                key_list = keys[0].split(':')
                v[key_list[-1]] = val
                for key in reversed(key_list[:-1]):
                    v = {key: v}
                # v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)

        return variants

    def variants(self):
        """
        Makes a list of dicts, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take
        the form

            ``'full:param:name'``.

        The colons are taken to indicate that these parameters should
        have a nested dict structure. eg, if there are two params,

            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:a'``    1
            ``'base:param:b'``    2
            ====================  ===

        the variant dict will have the structure

        .. parsed-literal::

            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }
                }
        """
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            """
            Build the full nested dict version of var, based on key names.
            """
            new_var = dict()
            unflatten_set = set()

            for k, v in var.items():
                if ':' in k:
                    splits = k.split(':')
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(
                        new_var[k0], dict
                    ), "You can't assign multiple values to the same key."

                    if not (k0 in new_var):
                        new_var[k0] = dict()

                    sub_k = ':'.join(splits[1:])
                    new_var[k0][sub_k] = v
                    unflatten_set.add(k0)
                else:
                    assert not (k in new_var), "You can't assign multiple values to the same key."
                    new_var[k] = v

            # Make sure to fill out the nested dicts.
            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])

            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]

        return new_variants

    def run(self, thunk, num_pool=1, data_dir=None):
        """
        Run each variant in the grid with function 'thunk'.

        Note: 'thunk' must be either a callable function, or a string. If it is
        a string, it must be the name of a parameter whose values are all
        callable functions.

        Uses ``call_experiment`` to actually launch each experiment, and gives
        each variant a name using ``self.variant_name()``.

        Maintenance note: the args for ExperimentGrid.run should track closely
        to the args for call_experiment. However, ``seed`` is omitted because
        we presume the user may add it as a parameter in the grid.
        """

        # Print info about self.
        self.print()

        # Make the list of all variants.
        variants = self.variants()

        # Print variant names for the user.
        var_names = set([self.variant_name(var) for var in variants])
        var_names = sorted(list(var_names))
        line = '=' * self.div_line_width
        preparing = colorize(
            'Preparing to run the following experiments...', color='green', bold=True
        )
        joined_var_names = '\n'.join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)

        if self.WAIT_BEFORE_LAUNCH > 0:
            delay_msg = (
                colorize(
                    dedent(
                        """
            Launch delayed to give you a few seconds to review your experiments.

            To customize or disable this behavior, change WAIT_BEFORE_LAUNCH in
            spinup/user_config.py.

            """
                    ),
                    color='cyan',
                    bold=True,
                )
                + line
            )
            print(delay_msg)
            wait, steps = self.WAIT_BEFORE_LAUNCH, 100
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

        # pool = multiprocessing.Pool(processes=num_pool)
        pool = Pool(max_workers=num_pool)
        # Run the variants.
        results = []
        exp_names = []
        for idx, var in enumerate(variants):
            exp_name = '_'.join([k + '_' + str(v) for k, v in var.items()])
            exp_names.append(exp_name)
            data_dir = os.path.join('./', 'exp-x', self.name, exp_name, '')
            var['data_dir'] = data_dir
            pool.submit(thunk, idx, var['algo'], var['env_id'], var)
        pool.shutdown()

        path = os.path.join('./', 'exp-x', self.name, 'exp-x-results.txt')
        str_len = max([len(exp_name) for exp_name in exp_names])
        exp_names = [exp_name.ljust(str_len) for exp_name in exp_names]
        with open(path, 'a+') as f:
            for idx, var in enumerate(variants):
                f.write(exp_names[idx] + ': ')
                reward, cost, ep_len = results[idx].get()
                f.write("reward:" + str(round(reward, 2)) + ',')
                f.write("cost:" + str(round(cost, 2)) + ',')
                f.write("ep_len:" + str(ep_len))
                f.write('\n')
