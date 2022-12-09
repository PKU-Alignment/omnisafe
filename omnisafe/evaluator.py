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

import json
import os

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.save_video import save_video

from omnisafe.algorithms.env_wrapper import EnvWrapper
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.utils.online_mean_std import OnlineMeanStd


class Evaluator:
    """This class includes common evaluation methods for safe RL algorithms."""

    def __init__(self, env=None, pi=None, obs_oms=None, play=True, save_replay=True):
        """Initialize the evaluator.
        Args:
            env (gymnasium.Env): the environment. if None, the environment will be created from the config.
            pi (omnisafe.algos.models.actor.Actor): the policy. if None, the policy will be created from the config.
            obs_oms (omnisafe.algos.models.obs_oms.ObsOMS): the observation OMS. Only used if obs_oms is not None.
        """
        # set the attributes
        self.env = env
        self.pi = pi
        self.obs_oms = obs_oms if obs_oms is not None else lambda x: x

        # set the render mode
        self.play = play
        self.save_replay = save_replay
        if play and save_replay:
            self.render_mode = 'rgb_array'
        elif play and not save_replay:
            self.render_mode = 'human'
        elif not play and save_replay:
            self.render_mode = 'rgb_array_list'
        else:
            self.render_mode = None

    def set_seed(self, seed):
        """Set the seed for the environment."""
        self.env.reset(seed=seed)

    def load_saved_model(self, save_dir: str, model_name: str):
        """Load a saved model.
        Args:
            save_dir (str): directory where the model is saved.
            model_name (str): name of the model.
        """
        # load the config
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('The config file is not found in the save directory.')

        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError('The model is not found in the save directory.')

        # make the environment
        env_id = cfg['env_id']
        self.env = EnvWrapper(env_id, render_mode=self.render_mode)

        # make the actor
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        if isinstance(action_space, Box):
            actor_fn = GaussianActor
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            actor_fn = CategoricalActor
            act_dim = action_space.n
        else:
            raise ValueError
        obs_dim = observation_space.shape[0]

        pi_cfg = cfg['cfgs']['model_cfgs']['ac_kwargs']['pi']
        weight_initialization_mode = cfg['cfgs']['model_cfgs']['weight_initialization_mode']
        self.pi = actor_fn(
            obs_dim=obs_dim,
            act_dim=act_dim,
            weight_initialization_mode=weight_initialization_mode,
            **pi_cfg,
        )
        self.pi.load_state_dict(model_params['pi'])

        # make the observation OMS
        if cfg['cfgs']['standardized_obs']:
            self.obs_oms = OnlineMeanStd(shape=observation_space.shape)
            self.obs_oms.load_state_dict(model_params['obs_oms'])
        else:
            self.obs_oms = lambda x: x

    def evaluate(self, num_episodes: int = 10, horizon: int = 1000, cost_criteria: float = 1.0):
        """Evaluate the agent for num_episodes episodes.
        Args:
            num_episodes (int): number of episodes to evaluate the agent.
            horizon (int): maximum number of steps per episode.
            cost_criteria (float): the cost criteria for the evaluation.
        Returns:
            episode_rewards (list): list of episode rewards.
            episode_costs (list): list of episode costs.
            episode_lengths (list): list of episode lengths.
        """
        if self.env is None or self.pi is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.'
            )

        episode_rewards = []
        episode_costs = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_ret, ep_cost = 0.0, 0.0

            for step in range(horizon):
                with torch.no_grad():
                    act, _ = self.pi.predict(
                        self.obs_oms(torch.as_tensor(obs, dtype=torch.float32)), deterministic=True
                    )
                obs, rew, cost, done, truncated, _ = self.env.step(act.numpy())
                ep_ret += rew
                ep_cost += (cost_criteria**step) * cost

                if done or truncated:
                    episode_rewards.append(ep_ret)
                    episode_costs.append(ep_cost)
                    episode_lengths.append(step + 1)
                    break

        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(episode_rewards):.3f}')
        print(f'Average episode cost: {np.mean(episode_costs):.3f}')
        print(f'Average episode length: {np.mean(episode_lengths):.3f}')
        return episode_rewards, episode_costs, episode_lengths

    def render(
        self,
        num_episode: int = 0,
        horizon: int = 1000,
        seed: int = None,
        play=True,
        save_replay_path: str = None,
    ):
        """Render the environment for one episode.
        Args:
            horizon (int): maximum number of steps per episode.
            seed (int): seed for the environment. If None, the environment will be reset with a random seed.
            save_replay_path (str): path to save the replay. If None, no replay is saved.
        """
        # remake the environment if the render mode can not support needed play or save_replay
        if self.env is None or self.pi is None:
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.'
            )

        if self.env.render_mode == None:
            print("Remake the environment with render_mode='rgb_array' to render the environment.")
            self.env = EnvWrapper(self.env.env_id, render_mode='rgb_array')
            self.render_mode = 'rgb_array'

        if self.env.render_mode == 'human' and save_replay_path is not None:
            print("Remake the environment with render_mode='rgb_array' to save the replay.")
            self.env = EnvWrapper(self.env.env_id, render_mode='rgb_array')
            self.render_mode = 'rgb_array'

        if self.env.render_mode == 'rgb_array_list' and play:
            print("Remake the environment with render_mode='rgb_array' to render the environment.")
            self.env = EnvWrapper(self.env.env_id, render_mode='rgb_array')
            self.render_mode = 'rgb_array'

        frames = []
        obs, _ = self.env.reset(seed=seed)

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            frames.append(self.env.render())

        step_starting_index = 0
        for episode_idx in range(num_episode):
            for step in range(horizon):
                with torch.no_grad():
                    act, _ = self.pi.predict(
                        self.obs_oms(torch.as_tensor(obs, dtype=torch.float32)), deterministic=True
                    )
                obs, _, _, done, truncated, _ = self.env.step(act.numpy())

                if self.render_mode == 'rgb_array':
                    frames.append(self.env.render())

                if done or truncated:
                    break

            if self.render_mode == 'rgb_array_list':
                frames = self.env.render()

            if save_replay_path is not None:
                save_video(
                    frames,
                    save_replay_path,
                    fps=self.env.metadata['render_fps'],
                    step_starting_index=step_starting_index,
                    episode_index=episode_idx,
                )
            step_starting_index = step + 1
            self.env.reset()
