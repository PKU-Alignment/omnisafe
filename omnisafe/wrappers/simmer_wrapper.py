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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY pid_kiND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Environment wrapper for Simmer algorithm."""

import copy

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.wrappers.on_policy_wrapper import OnPolicyEnvWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


class PidController:  # pylint: disable=too-many-instance-attributes
    """Using PID controller to control the safety budget in Simmer environment."""

    def __init__(
        self,
        cfgs,
        safety_budget: float = 25.0,
        lower_budget: float = 1.0,
        upper_budget: float = 25.0,
    ) -> None:
        """Initialize the PID controller.

        Args:
            cfgs (CfgNode): Configurations.
            safety_budget (float): The initial safety budget.
            lower_budget (float): The lower bound of safety budget.
            upper_budget (float): The upper bound of safety budget.
        """
        # PID parameters.
        self.pid_kp = cfgs.pid_kp
        self.pid_ki = cfgs.pid_ki
        self.pid_kd = cfgs.pid_kd

        # Low pass filter.
        self.tau = cfgs.tau

        # Initialize the PID controller.
        self.error = 0.0
        self.error_i = 0.0
        self.prev_action = 0
        self.prev_raw_action = 0
        self.step_size = cfgs.step_size

        # Set the initial safety budget.
        self.safety_budget = safety_budget
        self.lower_budget = lower_budget
        self.upper_budget = upper_budget

    def compute_raw_action(self, obs: float):
        """Compute the raw action based on current obs.

        Args:
            obs (float): The current observation.

        Returns:
            float: The raw action.
        """

        # Low pass filter.
        error_p = self.tau * self.error + (1 - self.tau) * (self.safety_budget - obs)
        self.error_i += self.error
        error_d = self.pid_kd * (self.prev_action - self.prev_raw_action)

        # Compute PID error.
        curr_raw_action = self.pid_kp * error_p + self.pid_ki * self.error_i + self.pid_kd * error_d
        return curr_raw_action

    def act(self, obs: float):
        """Compute the safety budget based on the observation ``Jc``.

        Args:
            obs (float): The current observation.

        Returns:
            float: The safety budget.
        """
        curr_raw_action = self.compute_raw_action(obs)

        # Clip the raw action.
        curr_action = np.clip(curr_raw_action, -self.step_size, self.step_size)
        self.prev_action = curr_action
        self.prev_raw_action = curr_raw_action
        raw_budget = self.safety_budget + curr_action

        # Clip the safety budget.
        self.safety_budget = np.clip(raw_budget, self.lower_budget, self.upper_budget)

        return self.safety_budget


class QController:  # pylint: disable=too-many-instance-attributes
    """Using Q-learning to control the safety budget in Simmer environment."""

    def __init__(
        self,
        cfgs,
        safety_budget: float = 25.0,
        lower_budget: float = 1.0,
        upper_budget: float = 25.0,
    ) -> None:
        """ "
        Initialize the Q-learning controller.

        Args:
            cfgs (CfgNode): The config file.
            safety_budget (float): The initial safety budget.
            lower_budget (float): The lower bound of the safety budget.
            upper_budget (float): The upper bound of the safety budget.
        """

        # Set the initial safety budget.
        self.lower_budget = lower_budget
        self.upper_budget = upper_budget

        # Initialize the Q-learning controller.
        self.state_dim = cfgs.state_dim
        self.act_dim = cfgs.act_dim
        self.q_function = np.zeros((cfgs.state_dim, cfgs.act_dim))
        self.state_space = np.linspace(self.lower_budget, self.upper_budget, cfgs.state_dim)
        self.action_space = np.linspace(-1, 1, cfgs.act_dim, dtype=int)
        self.state = safety_budget
        self.init_idx = np.argwhere(self.state_space == self.state)
        self.action = 0
        self.step(self.action)

        # Set the Q-learning parameters.
        self.tau = cfgs.tau
        self.threshold = cfgs.threshold
        self.q_lr = cfgs.q_lr

        # Use epsilon greedy to explore the environment.
        self.epsilon = cfgs.epsilon

        # Initialize the observation (Cost value per epoch) buffer.
        self.prev_obs = copy.copy(self.state)
        self.filtered_obs_buffer = []
        self.filtered_obs = 0

    def get_state_idx(self, state: float):
        """Get the state index.

        Args:
            state (float): The current state.

        Returns:
            int: The state index.
        """
        state_idx = np.argwhere(self.state_space == state)[0][0]
        return state_idx

    def get_action_idx(self, action: float):
        """Get the action index.

        Args:
            action (float): The current action.

        Returns:
            int: The action index.
        """
        action_idx = np.argwhere(self.action_space == action)
        return action_idx

    def get_random_action(self):
        """Get the random action.

        Returns:
            float: The random action.
        """
        action_idx = np.random.randint(0, self.act_dim)
        return self.action_space[action_idx]

    def get_greedy_action(self, state: float):
        """Get the greedy action.

        Args:
            state (float): The current state(``cost_limit``).

        Returns:
            float: The greedy action.
        """
        state_idx = self.get_state_idx(state)
        action_idx = np.argmax(self.q_function[state_idx, :])
        action = self.action_space[action_idx]
        return action

    def update_q_function(self, state: float, action: float, reward: float, next_state: float):
        """Update the Q function using the Bellman equation.

        Args:
            state (float): The current state.
            action (float): The current action.
            reward (float): The reward.
            next_state (float): The next state.
        """
        state_idx = self.get_state_idx(state)
        action_idx = self.get_action_idx(action)
        next_state_idx = self.get_state_idx(next_state)
        self.q_function[state_idx, action_idx] = (1 - self.q_lr) * self.q_function[
            state_idx, action_idx
        ] + self.q_lr * (reward + self.tau * np.max(self.q_function[next_state_idx, :]))

    def step(self, action: float):
        """Step the environment.

        Args:
            action (float): The current action.
        """
        state_idx = self.get_state_idx(self.state)
        state_idx = np.clip(state_idx + action, 0, self.state_dim - 1, dtype=int)
        self.state = self.state_space[state_idx]
        return self.state

    def reward(self, state: float, action: float, obs: float):
        """Get the reward function based on whether the observation is within the threshold.

        Args:
            state (float): The current state.
            action (float): The current action.
            obs (float): The observation.

            Returns:
                float: The reward.
        """
        action_idx = self.get_action_idx(action)
        if int(self.threshold > obs - state and obs - state > -self.threshold):
            reward = np.array([-1, 1, 0.5])[action_idx]
        elif int(obs - state <= -self.threshold):
            reward = np.array([-1, 0, 2])[action_idx]
        elif int(obs - state >= self.threshold):
            reward = np.array([2, -1, -1])[action_idx]
        return reward[0]

    def act(self, obs: float):
        """Return the safety budget based on the observation.

        Args:
            obs (float): The observation.

        Returns:
            float: The safety budget.
        """
        prev_obs = self.filtered_obs
        self.filtered_obs = self.tau * prev_obs + (1 - self.tau) * obs
        self.filtered_obs_buffer.append(self.filtered_obs)
        state = self.state

        # Use epsilon greedy to explore the environment
        epsilon = np.random.random()
        if epsilon > self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_greedy_action(state)
        reward = self.reward(state, action, self.filtered_obs)
        next_state = self.step(action)
        safety_budget = next_state

        # Update the Q function
        self.update_q_function(state, action, reward, next_state)
        return safety_budget


@WRAPPER_REGISTRY.register
class SimmerEnvWrapper(OnPolicyEnvWrapper):  # pylint: disable=too-many-instance-attributes
    """Wrapper for the Simmer environment."""

    def __init__(self, env_id, cfgs, **env_kwargs) -> None:
        """Initialize the Simmer environment wrapper.

        Args:
            env_id (str): The environment id.
            cfgs (Config): The configuration.
            env_kwargs (dict): The additional parameters of environments.
        """
        super().__init__(env_id, **env_kwargs)

        self.unsafe_reward = cfgs.unsafe_reward
        self.simmer_gamma = cfgs.simmer_gamma
        if cfgs.scale_safety_budget:
            self.safety_budget = (
                cfgs.lower_budget
                * (1 - self.simmer_gamma**self.max_ep_len)
                / (1 - self.simmer_gamma)
                / np.float32(self.max_ep_len)
            )
            self.lower_budget = (
                cfgs.lower_budget
                * (1 - self.simmer_gamma**self.max_ep_len)
                / (1 - self.simmer_gamma)
                / np.float32(self.max_ep_len)
            )
            self.upper_budget = (
                cfgs.upper_budget
                * (1 - self.simmer_gamma**self.max_ep_len)
                / (1 - self.simmer_gamma)
                / np.float32(self.max_ep_len)
            )
        else:
            self.safety_budget = cfgs.lower_budget
            self.lower_budget = cfgs.lower_budget
            self.upper_budget = cfgs.upper_budget
        self.rel_safety_budget = self.safety_budget / self.upper_budget
        self.safety_obs = self.rel_safety_budget
        high = np.array(np.hstack([self.env.observation_space.high, np.inf]), dtype=np.float32)
        low = np.array(np.hstack([self.env.observation_space.low, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(high=high, low=low)
        if cfgs.simmer_controller == 'PID':
            self.controller = PidController(
                cfgs.controller_cfgs,
                safety_budget=self.safety_budget,
                lower_budget=self.lower_budget,
                upper_budget=self.upper_budget,
            )
        elif cfgs.simmer_controller == 'Q':
            self.controller = QController(
                cfgs.controller_cfgs,
                safety_budget=self.safety_budget,
                lower_budget=self.lower_budget,
                upper_budget=self.upper_budget,
            )
        else:
            raise NotImplementedError(
                f'Controller type {cfgs.simmer_controller} is not implemented.'
            )

    def augment_obs(self, obs: np.array, safety_obs: np.array):
        """Augmenting the obs with the safety obs, if needed.

        Args:
            obs (np.array): The observation.
            safety_obs (np.array): The safety observation.

        Returns:
            np.array: The augmented observation.
        """
        augmented_obs = np.hstack([obs, safety_obs])
        return augmented_obs

    def safety_step(self, cost: np.ndarray) -> np.ndarray:
        """Update the normalized safety obs.

        Args:
            cost (np.ndarray): The cost.

        Returns:
            np.ndarray: The normalized safety obs.
        """
        self.safety_obs -= cost / self.upper_budget
        self.safety_obs /= self.simmer_gamma
        return self.safety_obs

    def safety_reward(self, reward: np.ndarray, next_safety_obs: np.ndarray) -> np.ndarray:
        """Update the reward based on the safety obs.

        Args:
            reward (np.ndarray): The reward.
            next_safety_obs (np.ndarray): The next safety obs.

        Returns:
            np.ndarray: The updated reward.
        """
        reward = reward * (next_safety_obs > 0) + self.unsafe_reward * (next_safety_obs <= 0)
        return reward

    def reset(self, seed=None):
        """Reset environment.

        Args:
            seed (int): The seed.

        Returns:
            np.array: The augmented observation.
            dict: The info.
        """
        self.curr_o, info = self.env.reset(seed=seed)
        self.rel_safety_budget = self.safety_budget / self.upper_budget
        self.safety_obs = self.rel_safety_budget
        self.curr_o = self.augment_obs(self.curr_o, self.safety_obs)
        return self.curr_o, info

    def step(self, action):
        """Step environment.

        Args:
            action (np.array): The action.

        Returns:
            np.array: The augmented observation.
            np.array: The reward.
            np.array: The cost.
            bool: The terminated flag.
            bool: The truncated flag.
            dict: The info.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        next_safety_obs = self.safety_step(cost)
        info['true_reward'] = reward
        info['safety_obs'] = next_safety_obs
        reward = self.safety_reward(reward, next_safety_obs)
        augmented_obs = self.augment_obs(next_obs, next_safety_obs)

        return augmented_obs, reward, cost, terminated, truncated, info

    def set_budget(self, Jc):
        """Set the safety budget.

        Args:
            Jc (np.array): The safety budget.

        Returns:
            np.array: The safety budget.
        """
        self.safety_budget = self.controller.act(Jc)

    # pylint: disable-next=too-many-locals
    def roll_out(self, agent, buf, logger):
        """Collect data and store to experience buffer.

        Args:
            agent (Agent): The agent.
            buf (Buffer): The buffer.
            logger (Logger): The logger.

        Returns:
            float: The episode return.
            float: The episode cost.
            int: The episode length.
            float: The episode budget.
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
                            'Metrics/SafetyBudget': self.safety_budget,
                        }
                    )
                ep_ret, ep_costs, ep_len, ep_budget = 0.0, 0.0, 0, 0.0
                obs, _ = self.reset()
        # Update safety budget after each epoch.
        self.set_budget(logger.get_stats('Metrics/EpCost')[0])
