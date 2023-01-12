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
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from omnisafe.common.normalizer import Normalizer
from omnisafe.common.record_queue import RecordQueue
from omnisafe.typing import NamedTuple, Optional
from omnisafe.utils.tools import expand_dims
from omnisafe.wrappers.cmdp_wrapper import CMDPWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@dataclass
class RolloutLog:
    """Log for roll out."""

    ep_ret: np.ndarray = 0.0
    ep_costs: np.ndarray = 0.0
    ep_len: np.ndarray = 0.0
    ep_budget: np.ndarray = 0.0


@dataclass
class SimmerData:
    """Data for Simmer RL."""

    safety_budget: float = 0.0
    upper_budget: float = 0.0
    lower_budget: float = 0.0
    relative_budget: float = 0.0
    unsafe_reward: float = 0.0
    safety_obs: np.ndarray = None


@dataclass
class RolloutData:
    """Data for roll out."""

    local_steps_per_epoch: int = 0
    max_ep_len: int = 0
    use_cost: bool = False
    current_obs: np.ndarray = 0.0
    rollout_log: RolloutLog = None
    simmer_data: SimmerData = None


@dataclass
class PidData:
    """Data for PID controller."""

    pid_kp: float
    pid_ki: float
    pid_kd: float
    tau: float
    step_size: float


@dataclass
class QData:
    """Data for Q controller."""

    state_dim: int
    action_dim: int
    tau: float
    threshold: float
    learning_rate: float
    epsilon: float


@dataclass
class QTable:
    """Q table for Q controller."""

    action_space: np.ndarray
    q_function: np.ndarray
    state_space: np.ndarray


# pylint: disable-next=too-many-instance-attributes
class PidController:
    """Using PID controller to control the safety budget in Simmer environment."""

    def __init__(
        self,
        cfgs: NamedTuple,
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
        self.pid_data = PidData(
            pid_kp=cfgs.pid_kp,
            pid_ki=cfgs.pid_ki,
            pid_kd=cfgs.pid_kd,
            tau=cfgs.tau,
            step_size=cfgs.step_size,
        )
        self.simmer_data = SimmerData(
            safety_budget=safety_budget,
            upper_budget=upper_budget,
            lower_budget=lower_budget,
        )

        # Initialize the PID controller.
        self.error = 0.0
        self.error_i = 0.0
        self.prev_action = 0
        self.prev_raw_action = 0

    def compute_raw_action(self, obs: float) -> float:
        r"""Compute the raw action based on current obs.

        Detailedly, the raw action is computed by the PID controller.

        .. math::
            a = K_p e_p + K_i \int e_p dt + K_d \frac{de_p}{dt}

        where :math:`e_p` is the error of the PID controller.

        Args:
            obs (float): The current observation.
        """
        # Low pass filter.
        error_p = self.pid_data.tau * self.error + (1 - self.pid_data.tau) * (
            self.simmer_data.safety_budget - obs
        )
        self.error_i += self.error
        error_d = self.pid_data.pid_kd * (self.prev_action - self.prev_raw_action)

        # Compute PID error.
        curr_raw_action = (
            self.pid_data.pid_kp * error_p
            + self.pid_data.pid_ki * self.error_i
            + self.pid_data.pid_kd * error_d
        )
        return curr_raw_action

    def act(self, obs: float) -> float:
        """Compute the safety budget based on the observation ``Jc``, following the several steps:

        - Compute the raw action based on the observation ``Jc``.
        - Clip the raw action.
        - Compute the safety budget.

        Args:
            obs (float): The current observation.
        """
        curr_raw_action = self.compute_raw_action(obs)

        # Clip the raw action.
        curr_action = np.clip(curr_raw_action, -self.pid_data.step_size, self.pid_data.step_size)
        self.prev_action = curr_action
        self.prev_raw_action = curr_raw_action
        raw_budget = self.simmer_data.safety_budget + curr_action

        # Clip the safety budget.
        self.simmer_data.safety_budget = np.clip(
            raw_budget, self.simmer_data.lower_budget, self.simmer_data.upper_budget
        )

        return self.simmer_data.safety_budget


# pylint: disable-next=too-many-instance-attributes
class QController:
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
        self.safety_budget = safety_budget
        self.q_data = QData(
            state_dim=cfgs.state_dim,
            action_dim=cfgs.act_dim,
            tau=cfgs.tau,
            threshold=cfgs.threshold,
            learning_rate=cfgs.q_lr,
            epsilon=cfgs.epsilon,
        )
        self.q_table = QTable(
            action_space=np.linspace(-1, 1, cfgs.act_dim, dtype=int),
            q_function=np.zeros((cfgs.state_dim, cfgs.act_dim)),
            state_space=np.linspace(lower_budget, upper_budget, cfgs.state_dim),
        )
        self.action = 0
        self.step(self.action)

        # Initialize the observation (Cost value per epoch) buffer.
        self.prev_obs = copy.copy(self.safety_budget)
        self.filtered_obs_buffer = []
        self.filtered_obs = 0

    def get_state_idx(self, state: float) -> int:
        """Get the state index.

        Args:
            state (float): The current state.
        """
        state_idx = np.argwhere(self.q_table.state_space == state)[0][0]
        return state_idx

    def get_action_idx(self, action: float) -> int:
        """Get the action index.

        Args:
            action (float): The current action.
        """
        action_idx = np.argwhere(self.q_table.action_space == action)
        return action_idx

    def get_random_action(self) -> float:
        """Get the random action.

        Returns:
            float: The random action.
        """
        action_idx = np.random.randint(0, self.q_data.action_dim)
        return self.q_table.action_space[action_idx]

    def get_greedy_action(self, state: float) -> float:
        """Get the greedy action.

        Args:
            state (float): The current state(``cost_limit``).
        """
        state_idx = self.get_state_idx(state)
        action_idx = np.argmax(self.q_table.q_function[state_idx, :])
        action = self.q_table.action_space[action_idx]
        return action

    def update_q_function(
        self, state: float, action: float, reward: float, next_state: float
    ) -> None:
        """Update the Q function using the Bellman equation.

        Detailedly, the Q function is updated as follows:

        .. math::
            Q(s, a) = (1 - \\alpha) Q(s, a) + \\alpha (r + \\tau \\max_{a'} Q(s', a'))

        where :math:`s` is the current state, :math:`a` is the current action,
        :math:`r` is the reward, :math:`s'` is the next state,
        :math:`\\alpha` is the learning rate,
        and :math:`\\tau` is the discount factor.

        Args:
            state (float): The current state.
            action (float): The current action.
            reward (float): The reward.
            next_state (float): The next state.
        """
        state_idx = self.get_state_idx(state)
        action_idx = self.get_action_idx(action)
        next_state_idx = self.get_state_idx(next_state)
        self.q_table.q_function[state_idx, action_idx] = (
            1 - self.q_data.learning_rate
        ) * self.q_table.q_function[state_idx, action_idx] + self.q_data.learning_rate * (
            reward + self.q_data.tau * np.max(self.q_table.q_function[next_state_idx, :])
        )

    def step(self, action: float) -> float:
        """Step the environment.

        Args:
            action (float): The current action.
        """
        state_idx = self.get_state_idx(self.safety_budget)
        state_idx = np.clip(state_idx + action, 0, self.q_data.state_dim - 1, dtype=int)
        self.safety_budget = self.q_table.state_space[state_idx]
        return self.safety_budget

    def reward(self, state: float, action: float, obs: float) -> float:
        r"""Get the reward function based on whether the observation is within the threshold.

        Detailedly, the reward function is defined as follows:

        .. list-table::

            *   -   States
                -   Increase
                -   No change
                -   Decrease
            *   -   Unsafe
                -   -1
                -   -1
                -   2
            *   -   Safe
                -   0.5
                -   1
                -   -1
            *   -   Very Safe
                -   0.5
                -   1
                -   -1

        Args:
            state (float): The current state.
            action (float): The current action.
            obs (float): The observation.
        """
        action_idx = self.get_action_idx(action)
        if int(self.q_data.threshold > obs - state and obs - state > -self.q_data.threshold):
            reward = np.array([-1, 1, 0.5])[action_idx]
        elif int(obs - state <= -self.q_data.threshold):
            reward = np.array([-1, 0, 2])[action_idx]
        elif int(obs - state >= self.q_data.threshold):
            reward = np.array([2, -1, -1])[action_idx]
        return reward[0]

    def act(self, obs: float) -> float:
        """Compute the safety budget based on the observation ``Jc``, following the several steps:

        - Filter the observation using a low-pass filter.
        - Use epsilon greedy to explore the environment.
        - Update the Q function by calling :meth:`update_q_function`.
        - Return the safety budget.

        Args:
            obs (float): The current observation.

        """
        prev_obs = self.filtered_obs
        self.filtered_obs = self.q_data.tau * prev_obs + (1 - self.q_data.tau) * obs
        self.filtered_obs_buffer.append(self.filtered_obs)
        state = self.safety_budget

        # Use epsilon greedy to explore the environment
        epsilon = np.random.random()
        if epsilon > self.q_data.epsilon:
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
# pylint: disable-next=too-many-instance-attributes
class SimmerWrapper(CMDPWrapper):
    r"""SimmerEnvWrapper.

    Simmer is a safe RL algorithm that uses a safety budget to control the exploration of the RL agent.
    Similar to :class:`SauteEnvWrapper`, Simmer uses state augmentation to ensure safety.
    Additionally, Simmer uses PID controller and Q learning controller to control the safety budget.

    .. note::

        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    ``omnisafe`` provides two implementations of Simmer RL: :class:`PPOSimmer` and :class:`PPOLagSimmer`.

    References:

    - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
    - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
      David Mguni, Jun Wang, Haitham Bou-Ammar.
    - URL: https://arxiv.org/abs/2202.06558

    """

    def __init__(self, env_id, cfgs: Optional[NamedTuple] = None, **env_kwargs) -> None:
        """Initialize environment wrapper.
        Args:
            env_id (str): environment id.
            cfgs (collections.namedtuple): configs.
            env_kwargs (dict): The additional parameters of environments.
        """
        super().__init__(env_id, cfgs, **env_kwargs)
        if hasattr(self.env, '_max_episode_steps'):
            max_ep_len = self.env._max_episode_steps
        else:
            max_ep_len = 1000
        if cfgs.scale_safety_budget:
            safety_budget = (
                cfgs.lower_budget
                * (1 - cfgs.simmer_gamma**max_ep_len)
                / (1 - cfgs.simmer_gamma)
                / np.float32(max_ep_len)
            )
            lower_budget = (
                cfgs.lower_budget
                * (1 - cfgs.simmer_gamma**max_ep_len)
                / (1 - cfgs.simmer_gamma)
                / np.float32(max_ep_len)
            )
            upper_budget = (
                cfgs.upper_budget
                * (1 - cfgs.simmer_gamma**max_ep_len)
                / (1 - cfgs.simmer_gamma)
                / np.float32(max_ep_len)
            )
        else:
            safety_budget = cfgs.lower_budget
            lower_budget = cfgs.lower_budget
            upper_budget = cfgs.upper_budget
        self.rollout_data = RolloutData(
            0.0,
            max_ep_len,
            False,
            None,
            RolloutLog(
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
                np.zeros((self.cfgs.num_envs, 1)),
            ),
            SimmerData(
                safety_budget=safety_budget,
                upper_budget=upper_budget,
                lower_budget=lower_budget,
                relative_budget=safety_budget / upper_budget,
                unsafe_reward=cfgs.unsafe_reward,
                safety_obs=safety_budget / upper_budget,
            ),
        )
        high = np.array(np.hstack([self.observation_space.high, np.inf]), dtype=np.float32)
        low = np.array(np.hstack([self.observation_space.low, np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(high=high, low=low)
        self.obs_normalizer = (
            Normalizer(shape=(self.cfgs.num_envs, self.observation_space.shape[0]), clip=5)
            if self.cfgs.normalized_obs
            else None
        )
        self.record_queue = RecordQueue(
            'ep_ret', 'ep_cost', 'ep_len', 'ep_budget', maxlen=self.cfgs.max_len
        )
        if cfgs.simmer_controller == 'PID':
            self.controller = PidController(
                cfgs.controller_cfgs,
                safety_budget=self.rollout_data.simmer_data.safety_budget,
                lower_budget=self.rollout_data.simmer_data.lower_budget,
                upper_budget=self.rollout_data.simmer_data.upper_budget,
            )
        elif cfgs.simmer_controller == 'Q':
            self.controller = QController(
                cfgs.controller_cfgs,
                safety_budget=self.rollout_data.simmer_data.safety_budget,
                lower_budget=self.rollout_data.simmer_data.lower_budget,
                upper_budget=self.rollout_data.simmer_data.upper_budget,
            )
        else:
            raise NotImplementedError(
                f'Controller type {cfgs.simmer_controller} is not implemented.'
            )
        self.rollout_data.current_obs = self.reset()[0]

    def augment_obs(self, obs: np.array) -> np.array:
        """Augmenting the obs with the safety obs.

        Detailedly, the augmented obs is the concatenation of the original obs and the safety obs.
        The safety obs is the safety budget minus the cost divided by the safety budget.

        Args:
            obs (np.array): observation.
            safety_obs (np.array): safety observation.
        """
        augmented_obs = np.hstack([obs, self.rollout_data.simmer_data.safety_obs])
        return augmented_obs

    def safety_step(self, cost: np.ndarray, done: bool) -> np.ndarray:
        """Update the normalized safety obs.

        Args:
            cost (np.array): cost.
        """
        if done:
            self.rollout_data.simmer_data.safety_obs = np.ones(
                (self.cfgs.num_envs, 1), dtype=np.float32
            )
        else:
            self.rollout_data.simmer_data.safety_obs -= (
                cost / self.rollout_data.simmer_data.upper_budget
            )
            self.rollout_data.simmer_data.safety_obs /= self.cfgs.simmer_gamma

    def safety_reward(self, reward: np.ndarray) -> np.ndarray:
        """Update the reward.

        Args:
            reward (np.array): reward.
            next_safety_obs (np.array): next safety observation.
        """
        for idx, safety_obs in enumerate(self.rollout_data.simmer_data.safety_obs):
            if safety_obs <= 0:
                reward[idx] = self.rollout_data.simmer_data.unsafe_reward
        return reward

    def reset(self) -> Tuple[np.ndarray, Dict]:
        r"""Reset environment.

        .. note::
            The safety obs is initialized to ``rel_safety_budget``,
            which is the safety budget divided by the upper budget.
            The safety budget is controlled by the controller.

        Args:
            seed (int): seed for environment reset.
        """
        obs, info = self.env.reset()
        if self.cfgs.num_envs == 1:
            obs = expand_dims(obs)
            info = [info]
        self.rollout_data.simmer_data.relative_budget = (
            self.rollout_data.simmer_data.safety_budget / self.rollout_data.simmer_data.upper_budget
        )
        self.rollout_data.simmer_data.safety_obs = (
            self.rollout_data.simmer_data.relative_budget
            * np.ones((self.cfgs.num_envs, 1), dtype=np.float32)
        )
        obs = self.augment_obs(obs)
        return obs, info

    def step(self, action: np.array) -> tuple((np.array, np.array, np.array, bool, dict)):
        """Step environment.

        .. note::
            The safety obs is updated by the cost.
            The reward is updated by the safety obs.
            Detailedly, the reward is the original reward if the safety obs is greater than 0,
            otherwise the reward is the unsafe reward.

        Args:
            action (np.array): action.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action.squeeze())
        # next_obs, rew, done, info = env.step(act)
        if self.cfgs.num_envs == 1:
            next_obs, reward, cost, terminated, truncated, info = expand_dims(
                next_obs, reward, cost, terminated, truncated, info
            )
            self.safety_step(cost, done=terminated | truncated)
            if terminated | truncated:
                augmented_obs, info = self.reset()
            else:
                augmented_obs = self.augment_obs(next_obs)
        else:
            augmented_obs = self.augment_obs(next_obs)
        self.rollout_data.rollout_log.ep_ret += reward
        self.rollout_data.rollout_log.ep_costs += cost
        self.rollout_data.rollout_log.ep_len += np.ones(self.cfgs.num_envs)
        self.rollout_data.rollout_log.ep_budget += self.rollout_data.simmer_data.safety_obs
        reward = self.safety_reward(reward)
        return augmented_obs, reward, cost, terminated, truncated, info

    def set_budget(self, Jc):
        """Set the safety budget by the controller.

        Args:
            Jc (np.array): The safety budget.
        """
        self.rollout_data.simmer_data.safety_budget = self.controller.act(Jc)

    def rollout_log(
        self,
        logger,
        idx,
    ) -> None:
        """Log the information of the rollout."""
        self.record_queue.append(
            ep_ret=self.rollout_data.rollout_log.ep_ret[idx],
            ep_cost=self.rollout_data.rollout_log.ep_costs[idx],
            ep_len=self.rollout_data.rollout_log.ep_len[idx],
            ep_budget=self.rollout_data.rollout_log.ep_budget[idx],
        )
        avg_ep_ret, avg_ep_cost, avg_ep_len, avg_ep_budget = self.record_queue.get_mean(
            'ep_ret', 'ep_cost', 'ep_len', 'ep_budget'
        )
        logger.store(
            **{
                'Metrics/EpRet': avg_ep_ret,
                'Metrics/EpCost': avg_ep_cost,
                'Metrics/EpLen': avg_ep_len,
                'Metrics/EpBudget': avg_ep_budget,
                'Metrics/SafetyBudget': self.rollout_data.simmer_data.safety_budget,
            }
        )
        self.set_budget(avg_ep_cost)
        (
            self.rollout_data.rollout_log.ep_ret[idx],
            self.rollout_data.rollout_log.ep_costs[idx],
            self.rollout_data.rollout_log.ep_len[idx],
            self.rollout_data.rollout_log.ep_budget[idx],
        ) = (0.0, 0.0, 0.0, 0.0)
