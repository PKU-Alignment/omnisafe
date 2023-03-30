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
"""Implementation of the AlgoWrapper Class."""

from __future__ import annotations

import difflib
import os
import sys
from typing import Any

import psutil
import torch

from omnisafe.algorithms import ALGORITHM2TYPE, ALGORITHMS, registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.envs import support_envs
from omnisafe.evaluator import Evaluator
from omnisafe.utils import distributed
from omnisafe.utils.config import check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.plotter import Plotter
from omnisafe.utils.tools import recursive_check_config


class AlgoWrapper:
    """Algo Wrapper for algo."""

    def __init__(
        self,
        algo: str,
        env_id: str,
        train_terminal_cfgs: dict[str, Any] | None = None,
        custom_cfgs: dict[str, Any] | None = None,
    ) -> None:
        self.algo = algo
        self.env_id = env_id
        # algo_type will set in _init_checks()
        self.algo_type: str
        self.agent: BaseAlgo

        self.train_terminal_cfgs = train_terminal_cfgs
        self.custom_cfgs = custom_cfgs
        self._evaluator: Evaluator = None
        self._plotter: Plotter = None
        self.cfgs = self._init_config()
        self._init_checks()

    def _init_config(self):
        """Init config."""
        assert self.algo in ALGORITHMS['all'], (
            f"{self.algo} doesn't exist. "
            f"Did you mean {difflib.get_close_matches(self.algo, ALGORITHMS['all'], n=1)[0]}?"
        )
        self.algo_type = ALGORITHM2TYPE.get(self.algo, '')
        if self.algo_type is None or self.algo_type == '':
            raise ValueError(f'{self.algo} is not supported!')
        if self.algo_type in {'off-policy', 'model-based'} and self.train_terminal_cfgs is not None:
            assert (
                self.train_terminal_cfgs['parallel'] == 1
            ), 'off-policy or model-based only support parallel==1!'
        cfgs = get_default_kwargs_yaml(self.algo, self.env_id, self.algo_type)

        # update the cfgs from custom configurations
        if self.custom_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.custom_cfgs:
                self.custom_cfgs.pop('env_id')
            if 'algo' in self.custom_cfgs:
                self.custom_cfgs.pop('algo')
            # validate the keys of custom configuration
            recursive_check_config(self.custom_cfgs, cfgs)
            # update the cfgs from custom configurations
            cfgs.recurisve_update(self.custom_cfgs)
            # save configurations specified in current experiment
            cfgs.update({'exp_increment_cfgs': self.custom_cfgs})
        # update the cfgs from custom terminal configurations
        if self.train_terminal_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('env_id')
            if 'algo' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('algo')
            # validate the keys of train_terminal_cfgs configuration
            recursive_check_config(self.train_terminal_cfgs, cfgs.train_cfgs)
            # update the cfgs.train_cfgs from train_terminal configurations
            cfgs.train_cfgs.recurisve_update(self.train_terminal_cfgs)
            # save configurations specified in current experiment
            cfgs.recurisve_update({'exp_increment_cfgs': {'train_cfgs': self.train_terminal_cfgs}})

        # the exp_name format is PPO-{SafetyPointGoal1-v0}-
        exp_name = f'{self.algo}-{{{self.env_id}}}'
        cfgs.recurisve_update({'exp_name': exp_name, 'env_id': self.env_id, 'algo': self.algo})
        cfgs.train_cfgs.recurisve_update(
            {'epochs': cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.update_cycle},
        )
        return cfgs

    def _init_checks(self):
        """Init checks."""
        assert isinstance(self.algo, str), 'algo must be a string!'
        assert isinstance(self.cfgs.train_cfgs.parallel, int), 'parallel must be an integer!'
        assert self.cfgs.train_cfgs.parallel > 0, 'parallel must be greater than 0!'
        assert (
            isinstance(self.custom_cfgs, dict) or self.custom_cfgs is None
        ), 'custom_cfgs must be a dict!'
        assert self.env_id in support_envs(), (
            f"{self.env_id} doesn't exist. "
            f'Did you mean {difflib.get_close_matches(self.env_id, support_envs(), n=1)[0]}?'
        )

    def learn(self):
        """Agent Learning."""
        # Use number of physical cores as default.
        # If also hardware threading CPUs should be used
        # enable this by the use_number_of_threads=True
        physical_cores = psutil.cpu_count(logical=False)
        use_number_of_threads = bool(self.cfgs.train_cfgs.parallel > physical_cores)

        check_all_configs(self.cfgs, self.algo_type)
        device = self.cfgs.train_cfgs.device
        if device == 'cpu':
            torch.set_num_threads(self.cfgs.train_cfgs.torch_threads)
        else:
            torch.set_num_threads(1)
            torch.cuda.set_device(self.cfgs.train_cfgs.device)
        if distributed.fork(
            self.cfgs.train_cfgs.parallel,
            use_number_of_threads=use_number_of_threads,
            device=self.cfgs.train_cfgs.device,
        ):
            # Re-launches the current script with workers linked by MPI
            sys.exit()
        self.agent = registry.get(self.algo)(
            env_id=self.env_id,
            cfgs=self.cfgs,
        )
        ep_ret, ep_cost, ep_len = self.agent.learn()

        self._init_statistical_tools()

        return ep_ret, ep_len, ep_cost

    def _init_statistical_tools(self):
        """Init statistical tools."""
        self._evaluator = Evaluator()
        self._plotter = Plotter()

    def plot(self, smooth=1):
        """Plot the training curve.

        Args:
            smooth (int): window size, for smoothing the curve.
        """
        assert self._plotter is not None, 'Please run learn() first!'
        self._plotter.make_plots(
            [self.agent.logger.log_dir],
            None,
            'Steps',
            'Rewards',
            False,
            self.agent.cost_limit,
            smooth,
            None,
            None,
            'mean',
            self.agent.logger.log_dir,
        )

    def evaluate(self, num_episodes: int = 10, cost_criteria: float = 1.0):
        """Agent Evaluation.

        Args:
            num_episodes (int): number of episodes to evaluate.
            cost_criteria (float): the cost criteria to evaluate.
        """
        assert self._evaluator is not None, 'Please run learn() first!'
        for item in os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save')):
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                self._evaluator.load_saved(save_dir=self.agent.logger.log_dir, model_name=item.name)
                self._evaluator.evaluate(num_episodes=num_episodes, cost_criteria=cost_criteria)

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
        assert self._evaluator is not None, 'Please run learn() first!'
        for item in os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save')):
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                self._evaluator.load_saved(
                    save_dir=self.agent.logger.log_dir,
                    model_name=item.name,
                    render_mode=render_mode,
                    camera_name=camera_name,
                    width=width,
                    height=height,
                )
                self._evaluator.render(num_episodes=num_episodes)
