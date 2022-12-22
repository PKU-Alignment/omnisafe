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
"""Environment wrapper for saute algorithms."""

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.wrappers.on_policy_wrapper import OnPolicyEnvWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@WRAPPER_REGISTRY.register
class SauteEnvWrapper(OnPolicyEnvWrapper):
    r"""SauteEnvWrapper."""

    def __init__(
        self,
        env_id,
        cfgs,
        render_mode=None,
    ) -> None:
        r"""Initialize SauteEnvWrapper.

        Args:
            env_id (str): environment id.
            cfgs (dict): configuration dictionary.
            render_mode (str): render mode.

        """
        super().__init__(env_id, render_mode)

        self.unsafe_reward = cfgs.unsafe_reward
        self.saute_gamma = cfgs.saute_gamma
        if cfgs.scale_safety_budget:
            self.safety_budget = (
                cfgs.safety_budget
                * (1 - self.saute_gamma**self.max_ep_len)
                / (1 - self.saute_gamma)
                / np.float32(self.max_ep_len)
            )
        else:
            self.safety_budget = cfgs.safety_budget
        self.safety_obs = 1.0
        high = np.array(np.hstack([self.env.observation_space.high, np.inf]), dtype=np.float32)
        low = np.array(np.hstack([self.env.observation_space.low, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(high=high, low=low)

    def augment_obs(self, obs: np.array, safety_obs: np.array):
        r"""Augmenting the obs with the safety obs.

        Args:
            obs (np.array): observation.
            safety_obs (np.array): safety observation.

        Returns:
            augmented_obs (np.array): augmented observation.
        """
        augmented_obs = np.hstack([obs, safety_obs])
        return augmented_obs

    def safety_step(self, cost: np.ndarray) -> np.ndarray:
        r"""Update the normalized safety obs.

        Args:
            cost (np.array): cost.

        Returns:
            safety_obs (np.array): normalized safety observation.
        """
        self.safety_obs -= cost / self.safety_budget
        self.safety_obs /= self.saute_gamma
        return self.safety_obs

    def safety_reward(self, reward: np.ndarray, next_safety_obs: np.ndarray) -> np.ndarray:
        r"""Update the reward.

        Args:
            reward (np.array): reward.
            next_safety_obs (np.array): next safety observation.

        Returns:
            reward (np.array): updated reward.
        """
        reward = reward * (next_safety_obs > 0) + self.unsafe_reward * (next_safety_obs <= 0)
        return reward

    def reset(self, seed=None):
        r"""Reset environment.

        Args:
            seed (int): seed for environment reset.

        Returns:
            self.curr_o (np.array): current observation.
            info (dict): environment info.
        """
        self.curr_o, info = self.env.reset(seed=seed)
        self.safety_obs = 1.0
        self.curr_o = self.augment_obs(self.curr_o, self.safety_obs)
        return self.curr_o, info

    def step(self, action):
        r"""Step environment.

        Args:
            action (np.array): action.

        Returns:
            augmented_obs (np.array): augmented observation.
            reward (np.array): reward.
            cost (np.array): cost.
            terminated (bool): whether the episode is terminated.
            truncated (bool): whether the episode is truncated.
            info (dict): environment info.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        next_safety_obs = self.safety_step(cost)
        info['true_reward'] = reward
        info['safety_obs'] = next_safety_obs
        reward = self.safety_reward(reward, next_safety_obs)
        augmented_obs = self.augment_obs(next_obs, next_safety_obs)

        return augmented_obs, reward, cost, terminated, truncated, info

    # pylint: disable-next=too-many-locals
    def roll_out(self, agent, buf, logger):
        r"""Collect data and store to experience buffer.

        Args:
            agent (Agent): agent.
            buf (Buffer): buffer.
            logger (Logger): logger.

        Returns:
            ep_ret (float): episode return.
            ep_costs (float): episode costs.
            ep_len (int): episode length.
            ep_budget (float): episode budget.
        """
        obs, _ = self.reset()
        ep_ret, ep_costs, ep_len, ep_budget = 0.0, 0.0, 0, 0.0
        for step_i in range(self.local_steps_per_epoch):
            action, value, cost_value, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, done, truncated, info = self.step(action)
            ep_ret += info['true_reward']
            ep_costs += (self.cost_gamma**ep_len) * cost
            ep_len += 1
            ep_budget += self.safety_obs

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
                            'Metrics/EpBudget': ep_budget,
                        }
                    )
                ep_ret, ep_costs, ep_len, ep_budget = 0.0, 0.0, 0, 0.0
                obs, _ = self.reset()
