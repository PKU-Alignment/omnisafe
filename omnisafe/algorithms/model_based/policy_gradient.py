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
"""The Policy Gradient algorithm in model-based setting."""

import time
from copy import deepcopy

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.models import EnsembleDynamicsModel, VirtualEnv
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import core
from omnisafe.utils.distributed_utils import proc_id
from omnisafe.wrappers import wrapper_registry


@registry.register
class PolicyGradientModelBased:  # pylint: disable=too-many-instance-attributes
    """The Policy Gradient algorithm in Model-Based setting.

    References:
        Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        URL: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
    """

    def __init__(self, env_id, cfgs=None) -> None:
        self.env_id = env_id
        self.cfgs = deepcopy(cfgs)
        self.algo = self.__class__.__name__
        self.wrapper_type = self.cfgs.wrapper_type
        self.env = wrapper_registry.get(self.wrapper_type)(self.algo, self.env_id)

        self.device = torch.device(self.cfgs.device)
        self.cost_gamma = self.cfgs.cost_gamma
        # Set up logger and save configuration to disk
        # Get local parameters before logger instance to avoid unnecessary print
        self.logger = Logger(exp_name=cfgs.exp_name, data_dir=cfgs.data_dir, seed=cfgs.seed)
        self.logger.save_config(cfgs.todict())

        # Set seed
        seed = int(cfgs.seed)
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set env
        self.env.env.reset(seed=seed)
        self.env.set_eplen(int(self.cfgs.max_ep_len))

        # Initialize dynamics model
        self.dynamics = EnsembleDynamicsModel(
            self.algo,
            self.env.env_type,
            self.device,
            state_size=self.env.dynamics_state_size,
            action_size=self.env.action_space.shape[0],
            reward_size=1,
            cost_size=1,
            **self.cfgs.dynamics_cfgs,
        )
        self.virtual_env = VirtualEnv(self.algo, self.dynamics, self.env_id, self.device)

        # Initialize off-policy buffer
        self.off_replay_buffer = OffPolicyBuffer(
            obs_space=self.env.observation_space,
            act_space=self.env.action_space,
            size=self.cfgs.replay_size,
            batch_size=self.cfgs.batch_size,
            device=self.device,
        )

        if self.algo in ['MBPPOLag', 'SafeLOOP']:
            self.use_actor = True
            self.actor_critic = self.set_algorithm_specific_actor_critic()
        else:
            self.use_actor = False

        # Setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()

        self.logger.log('Start with training.')

    def learn(self):  # pylint: disable=too-many-locals
        """training the policy."""
        self.start_time = time.time()
        ep_len, ep_ret, ep_cost = 0, 0, 0
        state = self.env.reset()
        time_step = 0
        last_policy_update, last_dynamics_update, last_log = 0, 0, 0
        while time_step < self.cfgs.max_real_time_steps:
            # select action
            action, action_info = self.select_action(time_step, state, self.env)

            next_state, reward, cost, terminated, truncated, info = self.env.step(
                action, self.cfgs.action_repeat
            )

            time_step += info['step_num']
            ep_cost += (self.cost_gamma**ep_len) * cost
            ep_len += 1
            ep_ret += reward
            self.store_real_data(
                time_step,
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

            state = next_state
            if terminated or truncated:
                self.logger.store(
                    **{
                        'Metrics/EpRet': ep_ret,
                        'Metrics/EpLen': ep_len * self.cfgs.action_repeat,
                        'Metrics/EpCost': ep_cost,
                    }
                )
                ep_ret, ep_cost, ep_len = 0, 0, 0
                state = self.env.reset()
                self.algo_reset()

            if (
                time_step % self.cfgs.update_dynamics_freq < self.cfgs.action_repeat
                and time_step - last_dynamics_update >= self.cfgs.update_dynamics_freq
            ):
                self.update_dynamics_model()
                last_dynamics_update = time_step

            if (
                self.use_actor
                and time_step % self.cfgs.update_policy_freq < self.cfgs.action_repeat
                and time_step - last_policy_update >= self.cfgs.update_policy_freq
            ):
                self.update_actor_critic(time_step)
                last_policy_update = time_step

            # Evaluate episode
            if (
                time_step % self.cfgs.log_freq < self.cfgs.action_repeat
                and time_step - last_log >= self.cfgs.log_freq
            ) or time_step == self.cfgs.max_real_time_steps - 1:
                self.log(time_step)
                self.logger.torch_save(itr=time_step)
                last_log = time_step
        # Close opened files to avoid number of open files overflow
        self.logger.close()

    def log(self, time_step: int):
        """
        logging data
        """
        self.logger.log_tabular('TotalEnvSteps3', time_step)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')
        # Some child classes may add information to logs
        self.algorithm_specific_logs(time_step)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.dump_tabular()

    def select_action(self, time_step, state, env):  # pylint: disable=unused-argument
        """
        Select action when interact with real environment.

        Returns:
            action, action_info
        """
        if self.env.env_type == 'gym':
            state = env.generate_lidar(state)
        state_vec = np.array(state)
        state_tensor = torch.as_tensor(state_vec, device=self.device, dtype=torch.float32)
        action, val, cval, logp = self.actor_critic.step(state_tensor)
        action = np.nan_to_num(action)
        action_info = {'state_vec': state_vec, 'val': val, 'cval': cval, 'logp': logp}
        return action, action_info

    def algorithm_specific_logs(self, time_step):
        """
        Use this method to collect log information.
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for CPO, etc

        Returns:
            No return
        """

    def update_actor_critic(self, time_step):
        """
        Use this method to update actor and critic.

        Returns:
            No return
        """

    def set_algorithm_specific_actor_critic(self):
        """
        Use this method to initialize network.
        e.g. Initialize Soft Actor Critic

        Returns:
            Actor_critic
        """
        self.actor_critic = ConstraintActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            model_cfgs=self.cfgs.model_cfgs,
        ).to(self.device)
        # Set up optimizer for policy and value function

        self.actor_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.actor, learning_rate=self.cfgs.actor_lr
        )
        self.reward_critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.reward_critic, learning_rate=self.cfgs.critic_lr
        )
        self.cost_critic_optimizer = core.set_optimizer(
            'Adam', module=self.actor_critic.cost_critic, learning_rate=self.cfgs.critic_lr
        )

        return self.actor_critic

    def update_dynamics_model(self):
        """
        training the dynamics model

        Returns:
            No return
        """

    def algo_reset(self):
        """
        reset algo parameters

        Returns:
            No return
        """

    # pylint: disable-next=too-many-arguments
    def store_real_data(
        self,
        time_step,
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
