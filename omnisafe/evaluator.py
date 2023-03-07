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

import json
import os

import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.utils.save_video import save_video

from omnisafe.adapter.online_adapter import OnlineAdapter as EnvWrapper
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env=None,
        actor=None,
        obs_normalize=None,
        play=True,
        save_replay=True,
    ):
        """Initialize the evaluator.

        Args:
            env (gymnasium.Env): the environment. if None, the environment will be created from the config.
            pi (omnisafe.algos.models.actor.Actor): the policy. if None, the policy will be created from the config.
            obs_normalize (omnisafe.algos.models.obs_normalize): the observation Normalize.
        """
        # set the attributes
        self.env = env
        self.actor = actor
        self.obs_normalizer = obs_normalize if obs_normalize is not None else lambda x: x
        self.env_wrapper_class = type(env) if env is not None else None

        # used when load model from saved file.
        self.cfgs = None
        self.save_dir = None
        self.model_name = None
        self.algo_name = None
        self.model_params = None

        # set the render mode
        self.play = play
        self.save_replay = save_replay
        self.set_render_mode(play, save_replay)

    def set_render_mode(self, play: bool = True, save_replay: bool = True):
        """Set the render mode.

        Args:
            render_mode (str): render mode.
        """
        # set the render mode
        if play and save_replay:
            self.render_mode = 'rgb_array'
        elif play and not save_replay:
            self.render_mode = 'human'
        elif not play and save_replay:
            self.render_mode = 'rgb_array_list'
        else:
            self.render_mode = None

    # pylint: disable-next=too-many-locals
    def load_saved_model(self, save_dir: str, model_name: str):
        """Load a saved model.

        Args:
            save_dir (str): directory where the model is saved.
            model_name (str): name of the model.
        """
        # load the config
        self.save_dir = save_dir
        self.model_name = model_name
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'The config file is not found in the save directory.'
            ) from error
        self.cfgs = Config.dict2config(kwargs)

        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            self.model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        self.algo_name = self.cfgs['exp_name'].split('-')[0]
        # make the environment
        env_id = self.cfgs['env_id']
        self.env = self._make_env(env_id, render_mode=self.render_mode)

        # make the actor
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        act_space_type = 'discrete' if isinstance(action_space, Discrete) else 'continuous'
        actor_type = self.cfgs['model_cfgs']['actor_type']

        pi_cfg = self.cfgs['model_cfgs']['actor']
        weight_initialization_mode = self.cfgs['model_cfgs']['weight_initialization_mode']
        actor_builder = ActorBuilder(
            obs_space=observation_space,
            act_space=action_space,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=weight_initialization_mode,
        )
        if act_space_type == 'discrete':
            self.actor = actor_builder.build_actor('categorical')
        else:
            self.actor = actor_builder.build_actor(actor_type)
        self.actor.load_state_dict(self.model_params['pi'])

    # pylint: disable-next=too-many-locals
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
            episode_rewards (list): list of episode rewards.
            episode_costs (list): list of episode costs.
            episode_lengths (list): list of episode lengths.
        """
        if self.env is None or self.actor is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.'
            )

        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        horizon = 1000

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            ep_ret, ep_cost = 0.0, 0.0

            for step in range(horizon):
                with torch.no_grad():
                    act = self.actor.predict(
                        torch.as_tensor(obs, dtype=torch.float32),
                        deterministic=True,
                    )
                obs, rew, cost, _, _, _ = self.env.step(act)
                ep_ret += rew
                ep_cost += (cost_criteria**step) * cost
            episode_costs.append(ep_cost.numpy().mean())
            episode_rewards.append(ep_ret.numpy().mean())
            episode_lengths.append(step)
            print(f'Episode {episode+1} results:')
            print(f'Episode reward: {ep_ret.numpy().mean()}')
            print(f'Episode cost: {ep_cost.numpy().mean()}')
            print(f'Episode length: {step+1}')
        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(episode_rewards)}')
        print(f'Average episode cost: {np.mean(episode_costs)}')
        print(f'Average episode length: {np.mean(episode_lengths)+1}')
        return (
            episode_rewards,
            episode_costs,
        )

    def render(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
        self,
        num_episodes: int = 0,
        play=True,
        save_replay_path: str = None,
        camera_name: str = None,
        camera_id: str = None,
        width: int = None,
        height: int = None,
    ):
        """Render the environment for one episode.

        Args:
            seed (int): seed for the environment. If None, the environment will be reset with a random seed.
            save_replay_path (str): path to save the replay. If None, no replay is saved.
        """

        if save_replay_path is None:
            save_replay_path = os.path.join(self.save_dir, 'video', self.model_name.split('.')[0])

        # remake the environment if the render mode can not support needed play or save_replay
        if self.env is None or self.actor is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.'
            )
        self.set_render_mode(play, save_replay_path is not None)
        print(f'Render mode: {self.render_mode}')
        width = self.env.width if width is None else width
        height = self.env.height if height is None else height
        env_kwargs = {
            'env_id': self.cfgs['env_id'],
            'render_mode': self.render_mode,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }
        self.env = self._make_env(**env_kwargs)
        if self.cfgs['algo_cfgs']['obs_normalize']:
            self.env.load(self.model_params['obs_normalizer'])
        horizon = 1000
        frames = []
        obs, _ = self.env.reset()
        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            frames.append(self.env.render())
        for episode_idx in range(num_episodes):
            for _ in range(horizon):
                with torch.no_grad():
                    act = self.actor.predict(obs, deterministic=True)
                obs, _, _, done, truncated, _ = self.env.step(act.cpu().squeeze())
                if done[0] or truncated[0]:
                    break
                if self.render_mode == 'rgb_array':
                    frames.append(self.env.render())

            if self.render_mode == 'rgb_array_list':
                frames = self.env.render()
            if save_replay_path is not None:
                save_video(
                    frames,
                    save_replay_path,
                    fps=self.env.fps,
                    episode_trigger=lambda x: True,
                    video_length=horizon,
                    episode_index=episode_idx,
                    name_prefix='eval',
                )
            self.env.reset()
            frames = []

    def _make_env(self, env_id, **env_kwargs):
        """Make wrapped environment."""

        return EnvWrapper(
            env_id, self.cfgs.train_cfgs.vector_env_nums, self.cfgs.seed, self.cfgs, **env_kwargs
        )
