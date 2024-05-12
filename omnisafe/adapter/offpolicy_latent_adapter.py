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
"""OffPolicy Latent Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import OffPolicySequenceBuffer
from omnisafe.common.latent import CostLatentModel
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import (
    ActionRepeat,
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config
from omnisafe.utils.model import ObservationConcator


class OffPolicyLatentAdapter(OnlineAdapter):
    """OffPolicy Adapter on Latent Space for OmniSafe.

    :class:`OffPolicyLatentAdapter` is used to adapt the vision-based environment to the off-policy
    training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyLatentAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        assert self.action_space.shape
        self._observation_concator: ObservationConcator = ObservationConcator(
            self._cfgs.algo_cfgs.latent_dim_1 + self._cfgs.algo_cfgs.latent_dim_2,
            self.action_space.shape,
            self._cfgs.algo_cfgs.num_sequences,
            device=self._device,
        )
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()
        self.z1: torch.Tensor = torch.zeros(1)
        self.z2: torch.Tensor = torch.zeros(1)
        self._initialized: bool = False
        self._reset_sequence_queue = False

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. hint::
            OmniSafe supports the following wrappers:

        +-----------------+--------------------------------------------------------+
        | Wrapper         | Description                                            |
        +=================+========================================================+
        | TimeLimit       | Limit the time steps of the environment.               |
        +-----------------+--------------------------------------------------------+
        | AutoReset       | Reset the environment when the episode is done.        |
        +-----------------+--------------------------------------------------------+
        | ObsNormalize    | Normalize the observation.                             |
        +-----------------+--------------------------------------------------------+
        | RewardNormalize | Normalize the reward.                                  |
        +-----------------+--------------------------------------------------------+
        | CostNormalize   | Normalize the cost.                                    |
        +-----------------+--------------------------------------------------------+
        | ActionScale     | Scale the action.                                      |
        +-----------------+--------------------------------------------------------+
        | Unsqueeze       | Unsqueeze the step result for single environment case. |
        +-----------------+--------------------------------------------------------+


        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, device=self._device, time_limit=1000)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = ActionScale(self._env, device=self._device, low=-1.0, high=1.0)
        self._env = ActionRepeat(self._env, times=2, device=self._device)

        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def _wrapper_eval(
        self,
        obs_normalize: bool = True,
    ) -> None:
        """Wrapper the environment for evaluation.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert self._eval_env, 'Your environment for evaluation does not exist!'
        if self._env.need_time_limit_wrapper:
            assert (
                self._eval_env.max_episode_steps
            ), 'You must define max_episode_steps as an\
                \ninteger or cancel the use of the time_limit wrapper.'
            self._eval_env = TimeLimit(
                self._eval_env,
                time_limit=self._eval_env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = ActionRepeat(self._eval_env, times=2, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    @property
    def latent_space(self) -> Box:
        """Get the latent space."""
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._cfgs.algo_cfgs.latent_dim_1 + self._cfgs.algo_cfgs.latent_dim_2,),
        )

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        latent_model: CostLatentModel,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            latent_model (CostLatentModel): Latent model, including encoder and decoder.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        assert self._eval_env, 'Your environment for evaluation does not exist!'
        assert self.action_space.shape
        eval_observation_concator: ObservationConcator = ObservationConcator(
            self._cfgs.algo_cfgs.latent_dim_1 + self._cfgs.algo_cfgs.latent_dim_2,
            self.action_space.shape,
            self._cfgs.algo_cfgs.num_sequences,
            device=self._device,
        )
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)
            eval_observation_concator.reset_episode(obs)

            with torch.no_grad():
                feature = latent_model.encoder(eval_observation_concator.last_state)

            z1_mean, z1_std = latent_model.z1_posterior_init(feature)
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            z2_mean, z2_std = latent_model.z2_posterior_init(z1)
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            latent_obs = torch.cat((z1, z2), dim=-1).squeeze()

            done = False
            while not done:
                act = agent.step(latent_obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )

                eval_observation_concator.append(obs, act)

                with torch.no_grad():
                    feature = latent_model.encoder(eval_observation_concator.last_state)
                z1_mean, z1_std = latent_model.z1_posterior(
                    torch.cat(
                        [feature.squeeze(), z2.squeeze(), eval_observation_concator.last_action],
                        dim=-1,
                    ),
                )
                z1 = z1_mean + torch.randn_like(z1_std) * z1_std
                z2_mean, z2_std = latent_model.z2_posterior(
                    torch.cat(
                        [z1.squeeze(), z2.squeeze(), eval_observation_concator.last_action],
                        dim=-1,
                    ),
                )
                z2 = z2_mean + torch.randn_like(z2_std) * z2_std
                latent_obs = torch.cat((z1, z2), dim=-1).squeeze()

                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += info.get('num_step', 1)

                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def pre_process(
        self,
        latent_model: CostLatentModel,
        concated_obs: ObservationConcator,
    ) -> torch.Tensor:
        """Processes the concatenated observations to produce latent representation.

        Args:
            latent_model (CostLatentModel): The latent model containing the encoder and decoder.
            concated_obs (ObservationConcator): An object that encapsulates the concatenated observations.

        Returns:
            A tensor combining the latent variables z1 and z2, representing the current state of
                the system in the latent space.
        """
        with torch.no_grad():
            feature = latent_model.encoder(concated_obs.last_state)

        if not self._initialized:
            z1_mean, z1_std = latent_model.z1_posterior_init(feature)
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            z2_mean, z2_std = latent_model.z2_posterior_init(self.z1)
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std
            self._initialized = True
        else:
            z1_mean, z1_std = latent_model.z1_posterior(
                torch.cat([feature.squeeze(), self.z2.squeeze(), concated_obs.last_action], dim=-1),
            )
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            z2_mean, z2_std = latent_model.z2_posterior(
                torch.cat([self.z1.squeeze(), self.z2.squeeze(), concated_obs.last_action], dim=-1),
            )
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        return torch.cat([self.z1, self.z2], dim=-1).squeeze()

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        latent_model: CostLatentModel,
        buffer: OffPolicySequenceBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            latent_model (CostLatentModel): Latent model, including encoder and decoder.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for step in range(rollout_step):
            if not self._reset_sequence_queue:
                buffer.reset_sequence_queue(self._current_obs)
                self._observation_concator.reset_episode(self._current_obs)
                self._reset_sequence_queue = True

            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).to(self._device)  # type: ignore
            else:
                act = agent.step(
                    self.pre_process(latent_model, self._observation_concator),
                    deterministic=False,
                )

            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            step += info.get('num_step', 1) - 1

            real_next_obs = next_obs.clone()

            self._observation_concator.append(next_obs, act)

            self._log_value(reward=reward, cost=cost, info=info)

            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
                    self.z1 = torch.zeros(1)
                    self.z2 = torch.zeros(1)
                    self._initialized = False
                    self._reset_sequence_queue = False
                if 'final_observation' in info:
                    real_next_obs[idx] = info['final_observation'][idx]

            buffer.store(
                obs=real_next_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
            )

            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += info.get('num_step', 1)

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._observation_concator.reset_episode(obs)
        return obs, info
