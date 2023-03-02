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
"""Implementation of the AlgoWrapper Class."""

import difflib
import sys
from typing import Any, Dict, Optional

import psutil
import torch
from safety_gymnasium.utils.registration import safe_registry

from omnisafe.algorithms import ALGORITHM2TYPE, ALGORITHMS, registry
from omnisafe.utils import distributed
from omnisafe.utils.config import get_default_kwargs_yaml
from omnisafe.utils.tools import terminal_cfgs_to_dict


class AlgoWrapper:
    """Algo Wrapper for algo."""

    def __init__(
        self,
        algo: str,
        env_id: str,
        parallel: int = 1,
        custom_cfgs: Optional[Dict[str, Any]] = None,
    ):
        self.algo = algo
        self.parallel = parallel
        self.env_id = env_id
        # algo_type will set in _init_checks()
        self.algo_type: str
        self.custom_cfgs = custom_cfgs
        self.evaluator = None
        self._init_checks()

    def _init_checks(self):
        """Init checks."""
        assert isinstance(self.algo, str), 'algo must be a string!'
        assert isinstance(self.parallel, int), 'parallel must be an integer!'
        assert self.parallel > 0, 'parallel must be greater than 0!'
        assert (
            isinstance(self.custom_cfgs, dict) or self.custom_cfgs is None
        ), 'custom_cfgs must be a dict!'
        assert self.algo in ALGORITHMS['all'], (
            f"{self.algo} doesn't exist. "
            f"Did you mean {difflib.get_close_matches(self.algo, ALGORITHMS['all'], n=1)[0]}?"
        )
        assert self.env_id in safe_registry, (
            f"{self.env_id} doesn't exist. "
            f'Did you mean {difflib.get_close_matches(self.env_id, safe_registry, n=1)[0]}?'
        )
        self.algo_type = ALGORITHM2TYPE.get(self.algo, '')
        if self.algo_type is None or self.algo_type == '':
            raise ValueError(f'{self.algo} is not supported!')
        if self.algo_type in ['off-policy', 'model-based']:
            assert self.parallel == 1, 'off-policy or model-based only support parallel==1!'

    def learn(self):
        """Agent Learning."""
        # Use number of physical cores as default.
        # If also hardware threading CPUs should be used
        # enable this by the use_number_of_threads=True
        physical_cores = psutil.cpu_count(logical=False)
        use_number_of_threads = bool(self.parallel > physical_cores)

        cfgs = get_default_kwargs_yaml(self.algo, self.env_id, self.algo_type)

        if self.custom_cfgs is not None:
            for key, val in self.custom_cfgs.items():
                custom_entity = terminal_cfgs_to_dict(key, val)
                cfgs.recurisve_update(custom_entity)

        # the exp_name format is PPO-<SafetyPointGoal1-v0>-
        exp_name = f'{self.algo}-<{self.env_id}>'
        cfgs.recurisve_update({'exp_name': exp_name, 'env_id': self.env_id})
        cfgs.train_cfgs.recurisve_update(
            {'epochs': cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.update_cycle}
        )

        # check_all_configs(cfgs, self.algo_type)
        torch.set_num_threads(cfgs.train_cfgs.torch_threads)
        if distributed.fork(
            self.parallel,
            use_number_of_threads=use_number_of_threads,
            device=cfgs.train_cfgs.device,
        ):
            # Re-launches the current script with workers linked by MPI
            sys.exit()
        agent = registry.get(self.algo)(
            env_id=self.env_id,
            cfgs=cfgs,
        )
        ep_ret, ep_cost, ep_len = agent.learn()
        return ep_ret, ep_len, ep_cost

    # def evaluate(self, num_episodes: int = 10, horizon: int = 1000, cost_criteria: float = 1.0):
    #     """Agent Evaluation."""
    #     assert self.evaluator is not None, 'Please run learn() first!'
    #     self.evaluator.evaluate(num_episodes, horizon, cost_criteria)

    # # pylint: disable-next=too-many-arguments
    # def render(
    #     self,
    #     num_episode: int = 0,
    #     horizon: int = 1000,
    #     seed: int = None,
    #     play=True,
    #     save_replay_path: Optional[str] = None,
    # ):
    #     """Render the environment."""
    #     assert self.evaluator is not None, 'Please run learn() first!'
    #     self.evaluator.render(num_episode, horizon, seed, play, save_replay_path)
