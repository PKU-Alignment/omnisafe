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

import safety_gymnasium
import torch

from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


# pylint: disable=too-many-instance-attributes
@WRAPPER_REGISTRY.register
class OffPolicyEnvWrapper:
    """OffPolicyEnvWrapper"""

    def __init__(
        self,
        env_id,
        use_cost,
        max_ep_len,
        render_mode=None,
    ):
        # check env_id is str
        self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env_id = env_id
        self.render_mode = render_mode
        self.metadata = self.env.metadata
        self.use_cost = use_cost

        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env._max_episode_steps
        else:
            self.max_ep_len = max_ep_len
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = None
        self.curr_o, _ = self.env.reset(seed=self.seed)
        self.ep_ret = 0
        self.ep_cost = 0
        self.ep_len = 0
        # self.deterministic = False
        self.local_steps_per_epoch = None
        self.cost_gamma = None
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

    # pylint: disable=too-many-arguments, too-many-locals
    def roll_out(
        self,
        actor_critic,
        buf,
        logger,
        deterministic,
        use_rand_action,
        ep_steps,
    ):
        """collect data and store to experience buffer."""
        for _ in range(ep_steps):
            ep_ret = self.ep_ret
            ep_len = self.ep_len
            ep_cost = self.ep_cost
            obs = self.curr_o
            action, value, cost_value, _ = actor_critic.step(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic
            )
            # Store values for statistic purpose
            if self.use_cost:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})
            if use_rand_action:
                action = self.env.action_space.sample()
            # Step the env
            # pylint: disable=unused-variable
            obs_next, reward, cost, done, truncated, info = self.step(action)
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            self.ep_len = ep_len
            self.ep_ret = ep_ret
            self.ep_cost = ep_cost
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            self.curr_o = obs_next
            if not deterministic:
                done = False if ep_len >= self.max_ep_len else done
                buf.store(obs, action, reward, cost, obs_next, done)
                if done or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCosts': ep_cost,
                        }
                    )
                    self.curr_o, _ = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_cost, self.ep_len = 0, 0, 0

            else:
                if done or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Test/EpRet': ep_ret,
                            'Test/EpLen': ep_len,
                            'Test/EpCosts': ep_cost,
                        }
                    )
                    self.curr_o, _ = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_cost, self.ep_len = 0, 0, 0

                    return
