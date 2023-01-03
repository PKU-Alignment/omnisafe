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
"""Implementation of Evaluator."""

import json
import os

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.save_video import save_video

from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config_utils import dict2namedtuple
from omnisafe.utils.online_mean_std import OnlineMeanStd
from omnisafe.wrappers.on_policy_wrapper import OnPolicyEnvWrapper as EnvWrapper
from omnisafe.wrappers.saute_wrapper import SauteEnvWrapper
from omnisafe.wrappers.simmer_wrapper import SimmerEnvWrapper


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env=None,
        actor=None,
        obs_oms=None,
        play=True,
        save_replay=True,
    ):
        """Initialize the evaluator.
        Args:
            env (gymnasium.Env): the environment. if None, the environment will be created from the config.
            pi (omnisafe.algos.models.actor.Actor): the policy. if None, the policy will be created from the config.
            obs_oms (omnisafe.algos.models.obs_oms.ObsOMS): the observation OMS. Only used if obs_oms is not None.
        """
        # set the attributes
        self.env = env
        self.actor = actor
        self.obs_oms = obs_oms if obs_oms is not None else lambda x: x
        self.env_wrapper_class = type(env) if env is not None else None

        # Used when load model from saved file.
        self.cfg = None
        self.save_dir = None
        self.model_name = None
        self.algo_name = None

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
            with open(cfg_path, encoding='utf-8', mode='r') as file:
                self.cfg = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'The config file is not found in the save directory.'
            ) from error

        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        self.algo_name = self.cfg['exp_name'].split('/')[1]

        # make the environment
        env_id = self.cfg['env_id']
        self.env = self._make_env(env_id, render_mode=self.render_mode)

        # make the actor
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        act_space_type = 'discrete' if isinstance(action_space, Discrete) else 'continuous'
        actor_type = self.cfg['model_cfgs']['ac_kwargs']['pi']['actor_type']
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n
        else:
            raise ValueError

        obs_dim = observation_space.shape[0]
        pi_cfg = self.cfg['model_cfgs']['ac_kwargs']['pi']
        weight_initialization_mode = self.cfg['model_cfgs']['weight_initialization_mode']
        actor_builder = ActorBuilder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=weight_initialization_mode,
            shared=None,
        )
        if act_space_type == 'discrete':
            self.actor = actor_builder.build_actor('categorical')
        else:
            act_max = torch.as_tensor(action_space.high)
            act_min = torch.as_tensor(action_space.low)
            self.actor = actor_builder.build_actor(actor_type, act_max=act_max, act_min=act_min)
        self.actor.load_state_dict(model_params['pi'])

        # make the observation OMS
        if self.cfg['standardized_obs']:
            self.obs_oms = OnlineMeanStd(shape=observation_space.shape)
            self.obs_oms.load_state_dict(model_params['obs_oms'])
        else:
            self.obs_oms = lambda x: x

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
        horizon = self.env.max_ep_len

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_ret, ep_cost = 0.0, 0.0

            for step in range(horizon):
                with torch.no_grad():
                    act = self.actor.predict(
                        self.obs_oms(torch.as_tensor(obs, dtype=torch.float32)),
                        deterministic=True,
                        need_log_prob=False,
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

    def render(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
        self,
        num_episode: int = 0,
        seed: int = None,
        play=True,
        save_replay_path: str = None,
        camera_name: str = None,
        camera_id: str = None,
        width: int = None,
        height: int = None,
        cost_criteria: float = 1.0,
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

        width = self.env.width if width is None else width
        height = self.env.height if height is None else height
        env_kwargs = {
            'env_id': self.env.env_id,
            'render_mode': 'rgb_array',
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }
        if self.env.render_mode is None:
            print("Remake the environment with render_mode='rgb_array' to render the environment.")
            self.env = self._make_env(**env_kwargs)
            self.render_mode = 'rgb_array'

        if self.env.render_mode == 'human' and save_replay_path is not None:
            print("Remake the environment with render_mode='rgb_array' to save the replay.")
            self.env = self._make_env(**env_kwargs)
            self.render_mode = 'rgb_array'

        if self.env.render_mode == 'rgb_array_list' and play:
            print("Remake the environment with render_mode='rgb_array' to render the environment.")
            self.env = self._make_env(**env_kwargs)
            self.render_mode = 'rgb_array'

        if self.env.camera_id != camera_id or self.env.camera_name != camera_name:
            print("Remake the environment with render_mode='rgb_array' to change the camera.")
            self.env = self._make_env(**env_kwargs)
            self.render_mode = 'rgb_array'

        if self.env.height != height or self.env.width != width:
            print(
                "Remake the environment with render_mode='rgb_array' to change the camera width or height."
            )
            self.env = self._make_env(**env_kwargs)
            self.render_mode = 'rgb_array'

        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        horizon = self.env.max_ep_len

        frames = []
        obs, _ = self.env.reset(seed=seed)
        ep_ret, ep_cost = 0.0, 0.0
        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            frames.append(self.env.render())

        if not os.path.exists(save_replay_path):
            os.makedirs(save_replay_path, exist_ok=True)
        with open(os.path.join(save_replay_path, 'log.txt'), mode='w', encoding='UTF-8') as log:
            for episode_idx in range(num_episode):
                for step in range(horizon):
                    with torch.no_grad():
                        act = self.actor.predict(
                            self.obs_oms(torch.as_tensor(obs, dtype=torch.float32)),
                            deterministic=True,
                            need_log_prob=False,
                        )
                    obs, rew, cost, done, truncated, _ = self.env.step(act.numpy())
                    ep_ret += rew
                    ep_cost += (cost_criteria**step) * cost

                    if self.render_mode == 'rgb_array':
                        frames.append(self.env.render())

                    if done or truncated:
                        episode_rewards.append(ep_ret)
                        episode_costs.append(ep_cost)
                        episode_lengths.append(step + 1)
                        print(
                            f'ep:{episode_idx} rew:{ep_ret} cost:{ep_cost} lenth:{step + 1}',
                            file=log,
                        )
                        break

                if self.render_mode == 'rgb_array_list':
                    frames = self.env.render()

                if save_replay_path is not None:
                    save_video(
                        frames,
                        save_replay_path,
                        fps=self.env.metadata['render_fps'],
                        episode_trigger=lambda x: True,
                        episode_index=episode_idx,
                        name_prefix='eval',
                    )
                obs, _ = self.env.reset()
                ep_ret, ep_cost = 0.0, 0.0
                frames = []

            print('Evaluation results:', file=log)
            print(f'Average episode reward: {np.mean(episode_rewards):.3f}', file=log)
            print(f'Average episode cost: {np.mean(episode_costs):.3f}', file=log)
            print(f'Average episode length: {np.mean(episode_lengths):.3f}', file=log)

    def _make_env(self, env_id, **env_kwargs):
        """Make wrapped environment."""
        if self.cfg is not None and 'env_cfgs' in self.cfg:
            env_cfgs = dict2namedtuple(self.cfg['env_cfgs'])

        if self.algo_name in ['PPOSimmerPid', 'PPOSimmerQ', 'PPOLagSimmerQ', 'PPOLagSimmerPid']:
            return SimmerEnvWrapper(env_id, env_cfgs, **env_kwargs)
        if self.algo_name in ['PPOSaute', 'PPOLagSaute']:
            return SauteEnvWrapper(env_id, env_cfgs, **env_kwargs)
        return EnvWrapper(env_id, **env_kwargs)
