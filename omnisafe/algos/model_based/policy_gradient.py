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
import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algos import registry
from omnisafe.algos.common.logger import Logger
from omnisafe.algos.common.replay_buffer import ReplayBuffer as Off_ReplayBuffer
from omnisafe.algos.model_based.models.dynamics_predict_env import PredictEnv
from omnisafe.algos.model_based.models.dynamicsmodel import EnsembleDynamicsModel
from omnisafe.algos.utils import distributed_utils
from omnisafe.algos.utils.distributed_utils import proc_id
from omnisafe.algos.utils.tools import get_flat_params_from


@registry.register
class PolicyGradientModelBased:
    """policy update base class"""

    def __init__(self, env, exp_name, data_dir, seed=0, algo='mbppo-lag', cfgs=None) -> None:
        self.env = env
        self.env_id = env.env_id
        self.cfgs = deepcopy(cfgs)
        self.exp_name = exp_name
        self.data_dir = data_dir
        self.algo = algo
        self.device = torch.device(self.cfgs['device'])

        # Set up logger and save configuration to disk
        # Get local parameters before logger instance to avoid unnecessary print
        self.params = locals()
        self.params.pop('self')
        self.params.pop('env')
        self.logger = Logger(exp_name=self.exp_name, data_dir=self.data_dir, seed=seed)
        self.logger.save_config(self.params)
        # Set seed
        seed = int(seed)
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set env
        self.env.env.reset(seed=seed)
        self.env.set_eplen(int(self.cfgs['max_ep_len']))

        # Init dynamics model
        reward_size = 1 if self.algo == 'safeloop' else 0
        self.dynamics = EnsembleDynamicsModel(
            algo,
            self.device,
            state_size=self.env.dynamics_state_size,
            action_size=self.env.action_space.shape[0],
            reward_size=reward_size,
            cost_size=0,
            **self.cfgs['dynamics_cfgs'],
        )
        self.predict_env = PredictEnv(algo, self.dynamics, self.env_id, self.device)

        # Init off-policy buffer
        # pylint: disable-next=line-too-long
        self.off_replay_buffer = Off_ReplayBuffer(
            self.env.dynamics_state_size,
            self.env.action_space.shape[0],
            self.cfgs['replay_size'],
            self.cfgs['batch_size'],
        )

        # Init Actor-Critic
        self.actor_critic = self.set_algorithm_specific_actor_critic()

        # Set up model saving
        what_to_save = {
            'pi': self.actor_critic.pi,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()
        # Setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()

        self.logger.log('Start with training.')

    def learn(self):
        """training the policy using safeloop"""
        self.start_time = time.time()
        ep_len, ep_ret, ep_cost = 0, 0, 0
        state = self.env.reset()
        t = 0
        while t < self.cfgs['max_real_time_steps']:
            # select action
            action, action_info = self.select_action(t, state, self.env)
            next_state, reward, cost, terminated, truncated, info = self.env.step(
                action, self.cfgs['action_repeat']
            )

            t += info['step_num']
            ep_len += 1
            ep_ret += reward
            ep_cost += cost

            self.store_real_data(
                t,
                ep_len,
                state,
                action_info,
                action,
                reward,
                cost,
                terminated,
                truncated,
                next_state,
                info,
            )

            if t % self.cfgs['update_dynamics_freq'] == 0:
                self.update_dynamics_model()

            if t % self.cfgs['update_policy_freq'] == 0:
                self.update_actor_critic(t)

            state = next_state
            if terminated or truncated:
                self.logger.store(
                    **{
                        'Metrics/EpRet': ep_ret,
                        'Metrics/EpLen': ep_len * self.cfgs['action_repeat'],
                        'Metrics/EpCosts': ep_cost,
                    }
                )
                ep_ret, ep_cost, ep_len = 0, 0, 0
                state = self.env.reset()
                self.algo_reset()

            # Evaluate episode
            if (t) % self.cfgs['log_freq'] == 0:
                self.log(t)
                self.logger.torch_save(itr=t)

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.actor_critic

    def log(self, timestep: int):
        """
        logging data
        """
        self.logger.log_tabular('TotalEnvSteps', timestep)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCosts')
        self.logger.log_tabular('Metrics/EpLen')
        # Some child classes may add information to logs
        self.algorithm_specific_logs(timestep)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.dump_tabular()

    def set_algorithm_specific_actor_critic(self):
        """
        Use this method to initialize network.
        e.g. Initialize Soft Actor Critic
        """

    def algorithm_specific_logs(self, timestep):
        """
        Use this method to collect log information.
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for CPO, etc
        """

    def update_dynamics_model(self):
        """
        training the dynamics model

        Returns:
            No return
        """

    def update_actor_critic(self, data=None):
        """
        update the actor critic

        Returns:
            No return
        """

    def algo_reset(self):
        """
        reset algo parameters

        Returns:
            No return
        """

    def store_real_data(
        self,
        timestep,
        ep_len,
        state,
        action_info,
        action,
        reward,
        cost,
        terminated,
        truncated,
        next_state,
        info,
    ):
        """
        store real env data to buffer

        Returns:
            No return
        """
