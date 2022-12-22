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
"""env_wrapper"""

import collections
from copy import deepcopy
from typing import Optional

import safety_gymnasium
import torch

from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@WRAPPER_REGISTRY.register
class OnPolicyEnvWrapper:  # pylint: disable=too-many-instance-attributes
    """env_wrapper"""

    def __init__(self, env_id, cfgs: Optional[collections.namedtuple] = None, render_mode=None):
        r"""Initialize environment wrapper.

        Args:
            env_id (str): environment id.
            cfgs (collections.namedtuple): configs.
            render_mode (str): render mode.
        """
        self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.cfgs = deepcopy(cfgs)
        self.env_id = env_id
        self.render_mode = render_mode
        self.metadata = self.env.metadata

        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env._max_episode_steps
        else:
            self.max_ep_len = 1000
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = None
        self.curr_o, _ = self.env.reset(seed=self.seed)
        self.rand_a = True
        self.ep_steps = 1000
        self.ep_ret = 0
        self.ep_costs = 0
        self.ep_len = 0
        self.deterministic = False
        self.local_steps_per_epoch = None
        self.cost_gamma = None
        self.use_cost = None
        self.penalty_param = None

    def make(self):
        """create environments"""
        return self.env

    def reset(self, seed=None):
        """reset environment"""
        self.curr_o, info = self.env.reset(seed=seed)
        return self.curr_o, info

    def render(self):
        """render environment"""
        return self.env.render()

    def set_seed(self, seed):
        """set environment seed"""
        self.seed = seed

    def set_rollout_cfgs(self, **kwargs):
        """set rollout configs"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def step(self, action):
        """engine step"""
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return next_obs, reward, cost, terminated, truncated, info

    # pylint: disable-next=too-many-locals
    def roll_out(self, agent, buf, logger):
        """collect data and store to experience buffer."""
        obs, _ = self.env.reset()
        ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
        for step_i in range(self.local_steps_per_epoch):
            action, value, cost_value, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, done, truncated, _ = self.step(action)
            ep_ret += reward
            ep_costs += (self.cost_gamma**ep_len) * cost
            ep_len += 1

            # Save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buffer
            buf.store(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=logp,
                cost=cost,
                cost_val=cost_value,
            )

            # Store values for statistic purpose
            if self.use_cost:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})

            # Update observation
            obs = next_obs

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout or truncated
            epoch_ended = step_i == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, value, cost_value, _ = agent(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value, cost_value = 0.0, 0.0

                # Automatically compute GAE in buffer
                buf.finish_path(value, cost_value, penalty_param=float(self.penalty_param))

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCost': ep_costs,
                        }
                    )
                ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
                obs, _ = self.env.reset()
