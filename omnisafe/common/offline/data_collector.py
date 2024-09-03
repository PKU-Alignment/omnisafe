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
"""Offline data collector for generating training data for offline algorithms."""

import json
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from gymnasium.spaces import Box
from tqdm import tqdm

from omnisafe.common.normalizer import Normalizer
from omnisafe.envs.core import make
from omnisafe.envs.wrapper import ActionScale
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config


@dataclass
class OfflineAgent:
    """A data class for storing the information of an agent."""

    agent_step: Callable[[torch.Tensor], torch.Tensor]
    size: int


class OfflineDataCollector:
    """A class for collecting offline data.

    Example:
        >>> # please change agent path and env name
        >>> env_name = 'SafetyPointCircle1-v0'
        >>> size = 2_000_000
        >>> agents = [
        >>>     ('./runs/PPO', 'epoch-500', 1_000_000),
        >>>     ('./runs/CPO', 'epoch-500', 1_000_000),
        >>> ]
        >>> save_dir = './data'

        >>> col = OfflineDataCollector(size, env_name)
        >>> for agent, model_name, size in agents:
        >>>     col.register_agent(agent, model_name, size)
        >>> col.collect(save_dir)
    """

    def __init__(self, size: int, env_name: str) -> None:
        """Initialize the data collector.

        Args:
            size (int): The total number of data to collect.
            env_name (str): The name of the environment.
        """
        self._size = size
        self._env_name = env_name

        # make a env, get the observation space and action space
        self._env = make(env_name)
        self._obs_space = self._env.observation_space
        self._action_space = self._env.action_space

        self._env = ActionScale(self._env, device=torch.device('cpu'), high=1.0, low=-1.0)

        if not isinstance(self._obs_space, Box):
            raise NotImplementedError('Only support Box observation space for now.')
        if not isinstance(self._action_space, Box):
            raise NotImplementedError('Only support Box action space for now.')

        # create a buffer to store the data
        self._obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)
        self._action = np.zeros((size, *self._action_space.shape), dtype=np.float32)
        self._reward = np.zeros((size, 1), dtype=np.float32)
        self._cost = np.zeros((size, 1), dtype=np.float32)
        self._next_obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)
        self._done = np.zeros((size, 1), dtype=np.float32)

        self.agents: List[OfflineAgent] = []

    def register_agent(self, save_dir: str, model_name: str, size: int) -> None:
        """Register an agent to the data collector.

        Args:
            save_dir (str): The directory of the agent.
            model_name (str): The name of the model.
            size (int): The number of data to collect from this agent.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'The config file is not found in the save directory.',
            ) from error
        cfgs = Config.dict2config(kwargs)

        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path, weights_only=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(f'Model {model_name} not found in {save_dir}') from error

        assert isinstance(self._obs_space, Box), 'Only support Box observation space for now.'
        if cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=self._obs_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
        else:

            def obs_normalizer(x: torch.Tensor) -> torch.Tensor:  # type: ignore
                return x

        actor_type = cfgs['model_cfgs']['actor_type']
        pi_cfg = cfgs['model_cfgs']['actor']
        weight_initialization_mode = cfgs['model_cfgs']['weight_initialization_mode']
        actor_builder = ActorBuilder(
            obs_space=self._obs_space,
            act_space=self._action_space,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=weight_initialization_mode,
        )
        actor = actor_builder.build_actor(actor_type)
        actor.load_state_dict(model_params['pi'])

        def agent_step(obs: torch.Tensor) -> torch.Tensor:
            obs = obs_normalizer(obs)
            return actor.predict(obs, deterministic=False)

        self.agents.append(OfflineAgent(agent_step, size))

    def collect(self, save_dir: str) -> None:
        """Collect data from the registered agents.

        Args:
            save_dir (str): The directory to save the collected data.
        """
        # check each agent's size
        total_size = 0
        for agent in self.agents:
            assert agent.size <= self._size, f'Agent {agent} size is larger than collector size.'
            total_size += agent.size
        assert total_size == self._size, 'Sum of agent size is not equal to collector size.'

        # collect data
        ptx = 0
        progress_bar = tqdm(total=self._size, desc='Collecting data...')

        for agent in self.agents:
            ep_ret, ep_cost, ep_len, single_ep_len, episode_num = 0.0, 0.0, 0.0, 0.0, 0.0

            obs, _ = self._env.reset()
            for _ in range(agent.size):
                action = agent.agent_step(obs)
                next_obs, reward, cost, terminate, truncated, _ = self._env.step(action)
                done = terminate or truncated

                self._obs[ptx] = obs.detach().numpy()
                self._action[ptx] = action.detach().numpy()
                self._reward[ptx] = reward.detach().numpy()
                self._cost[ptx] = cost.detach().numpy()
                self._next_obs[ptx] = next_obs.detach().numpy()
                self._done[ptx] = done.detach().numpy()

                ep_ret += reward.item()
                ep_cost += cost.item()
                ep_len += 1
                single_ep_len += 1

                ptx += 1
                obs = next_obs
                if done:
                    obs, _ = self._env.reset()
                    episode_num += 1
                    progress_bar.update(single_ep_len)
                    single_ep_len = 0

            print(f'Agent {agent} collected {agent.size} data points.')
            print(f'Average return: {ep_ret / episode_num}')
            print(f'Average cost: {ep_cost / episode_num}')
            print(f'Average episode length: {ep_len / episode_num}')
            print()

        # save data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{self._env_name}_data.npz')
        if os.path.exists(save_path):
            print(f'Warning: {save_path} already exists.')
            print(f'Warning: {save_path} will be overwritten.')
        np.savez(
            os.path.join(save_dir, f'{self._env_name}_data.npz'),
            obs=self._obs,
            action=self._action,
            reward=self._reward,
            cost=self._cost,
            next_obs=self._next_obs,
            done=self._done,
        )
