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
"""Implementation of Evaluator."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, make
from omnisafe.envs.wrapper import ActionScale, ObsNormalize, TimeLimit
from omnisafe.models.actor import ActorBuilder
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env: CMDP | None = None,
        actor: Actor | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            env (gymnasium.Env): the environment. if None, the environment will be created from the config.
            pi (omnisafe.algos.models.actor.Actor): the policy. if None, the policy will be created from the config.
            obs_normalize (omnisafe.algos.models.obs_normalize): the observation Normalize.
        """
        # set the attributes
        self._env: CMDP = env
        self._actor: Actor = actor

        # used when load model from saved file.
        self._cfgs: Config
        self._save_dir: str
        self._model_name: str

        self._dividing_line = '\n' + '#' * 50 + '\n'

        self.__set_render_mode(render_mode)

    def __set_render_mode(self, render_mode: str):
        """Set the render mode.

        Args:
            play (bool): whether to play the video.
            save_replay (bool): whether to save the video.
        """
        # set the render mode
        if render_mode in ['human', 'rgb_array', 'rgb_array_list', None]:
            self._render_mode = render_mode
        else:
            raise NotImplementedError('The render mode is not implemented.')

    def __load_cfgs(self, save_dir: str):
        """Load the config from the save directory.

        Args:
            save_dir (str): directory where the model is saved.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        self._cfgs = Config.dict2config(kwargs)

    def __load_model_and_env(self, save_dir: str, model_name: str, env_kwargs: dict[str, Any]):
        """Load the model from the save directory.

        Args:
            save_dir (str): directory where the model is saved.
            model_name (str): name of the model.
        """
        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        # load the environment
        self._env = make(**env_kwargs)

        observation_space = self._env.observation_space
        action_space = self._env.action_space

        assert isinstance(observation_space, Box), 'The observation space must be Box.'
        assert isinstance(action_space, Box), 'The action space must be Box.'

        if self._cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=observation_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
            self._env = ObsNormalize(self._env, device=torch.device('cpu'), norm=obs_normalizer)
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, device=torch.device('cpu'), time_limit=1000)
        self._env = ActionScale(self._env, device=torch.device('cpu'), low=-1.0, high=1.0)

        actor_type = self._cfgs['model_cfgs']['actor_type']
        pi_cfg = self._cfgs['model_cfgs']['actor']
        weight_initialization_mode = self._cfgs['model_cfgs']['weight_initialization_mode']
        actor_builder = ActorBuilder(
            obs_space=observation_space,
            act_space=action_space,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=weight_initialization_mode,
        )
        self._actor = actor_builder.build_actor(actor_type)
        self._actor.load_state_dict(model_params['pi'])

    # pylint: disable-next=too-many-locals
    def load_saved(
        self,
        save_dir: str,
        model_name: str,
        render_mode: str | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 256,
        height: int = 256,
    ):
        """Load a saved model.

        Args:
            save_dir (str): directory where the model is saved.
            model_name (str): name of the model.
        """
        # load the config
        self._save_dir = save_dir
        self._model_name = model_name

        self.__load_cfgs(save_dir)

        if render_mode is not None or self._render_mode is None:
            self.__set_render_mode(render_mode)

        env_kwargs = {
            'env_id': self._cfgs['env_id'],
            'num_envs': 1,
            'render_mode': self._render_mode,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }

        self.__load_model_and_env(save_dir, model_name, env_kwargs)

    def evaluate(
        self,
        num_episodes: int = 10,
        cost_criteria: float = 1.0,
    ):
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int): number of episodes to evaluate the agent.
            cost_criteria (float): the cost criteria for the evaluation.

        Returns:
            (float, float, float): the average return, the average cost, and the average length of the episodes.
        """
        if self._env is None or self._actor is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.',
            )

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []

        for episode in range(num_episodes):
            obs, _ = self._env.reset()
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0

            done = False
            while not done:
                with torch.no_grad():
                    act = self._actor.predict(
                        torch.as_tensor(obs, dtype=torch.float32),
                        deterministic=False,
                    )
                obs, rew, cost, terminated, truncated, _ = self._env.step(act)

                ep_ret += rew.item()
                ep_cost += (cost_criteria**length) * cost.item()
                length += 1

                done = bool(terminated or truncated)

            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)

            print(f'Episode {episode+1} results:')
            print(f'Episode reward: {ep_ret}')
            print(f'Episode cost: {ep_cost}')
            print(f'Episode length: {length}')

        print(self._dividing_line)
        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(episode_rewards)}')
        print(f'Average episode cost: {np.mean(episode_costs)}')
        print(f'Average episode length: {np.mean(episode_lengths)}')
        return (
            episode_rewards,
            episode_costs,
        )

    @property
    def fps(self) -> int:
        """The fps of the environment.

        Returns:
            int: the fps.
        """
        try:
            fps = self._env.metadata['render_fps']
        except AttributeError:
            fps = 30
            warnings.warn('The fps is not found, use 30 as default.', stacklevel=2)

        return fps

    def render(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
        self,
        num_episodes: int = 0,
        save_replay_path: str | None = None,
        max_render_steps: int = 2000,
        cost_criteria: float = 1.0,
    ):
        """Render the environment for one episode.

        Args:
            seed (int): seed for the environment. If None, the environment will be reset with a random seed.
            save_replay_path (str): path to save the replay. If None, no replay is saved.
        """

        if save_replay_path is None:
            save_replay_path = os.path.join(self._save_dir, 'video', self._model_name.split('.')[0])
        result_path = os.path.join(save_replay_path, 'result.txt')
        print(self._dividing_line)
        print(f'Saving the replay video to {save_replay_path},\n and the result to {result_path}.')
        print(self._dividing_line)

        horizon = 1000
        frames = []
        obs, _ = self._env.reset()
        if self._render_mode == 'human':
            self._env.render()
        elif self._render_mode == 'rgb_array':
            frames.append(self._env.render())

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []

        for episode_idx in range(num_episodes):
            step = 0
            done = False
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0
            while (
                not done and step <= max_render_steps
            ):  # a big number to make sure the episode will end
                with torch.no_grad():
                    act = self._actor.predict(obs, deterministic=False)
                obs, rew, cost, terminated, truncated, _ = self._env.step(act)
                step += 1
                done = bool(terminated or truncated)
                ep_ret += rew.item()
                ep_cost += (cost_criteria**length) * cost.item()
                length += 1

                if self._render_mode == 'rgb_array':
                    frames.append(self._env.render())

            if self._render_mode == 'rgb_array_list':
                frames = self._env.render()

            if save_replay_path is not None:
                save_video(
                    frames,
                    save_replay_path,
                    fps=self.fps,
                    episode_trigger=lambda x: True,
                    video_length=horizon,
                    episode_index=episode_idx,
                    name_prefix='eval',
                )
            self._env.reset()
            frames = []
            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)
            with open(result_path, 'a+', encoding='utf-8') as f:
                print(f'Episode {episode_idx+1} results:', file=f)
                print(f'Episode reward: {ep_ret}', file=f)
                print(f'Episode cost: {ep_cost}', file=f)
                print(f'Episode length: {length}', file=f)
        with open(result_path, 'a+', encoding='utf-8') as f:
            print(self._dividing_line)
            print('Evaluation results:', file=f)
            print(f'Average episode reward: {np.mean(episode_rewards)}', file=f)
            print(f'Average episode cost: {np.mean(episode_costs)}', file=f)
            print(f'Average episode length: {np.mean(episode_lengths)}', file=f)
