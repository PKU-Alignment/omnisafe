# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""CRABS Adapter for OmniSafe."""

from __future__ import annotations

import torch
from rich import errors
from rich.progress import track

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.control_barrier_function.crabs.models import MeanPolicy
from omnisafe.common.logger import Logger
from omnisafe.envs.crabs_env import CRABSEnv
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class CRABSAdapter(OffPolicyAdapter):
    """CRABS Adapter for OmniSafe.

    :class:`CRABSAdapter` is used to adapt the environment to the CRABS algorithm training.

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
        """Initialize a instance of :class:`CRABSAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._env: CRABSEnv
        self._eval_env: CRABSEnv
        self.n_expl_episodes = 0
        self._max_ep_len = self._env.env.spec.max_episode_steps  # type: ignore
        self.horizon = self._max_ep_len

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic | MeanPolicy,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorQCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/RawPolicyEpRet': ep_ret,
                    'Metrics/RawPolicyEpCost': ep_cost,
                    'Metrics/RawPolicyEpLen': ep_len,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorQCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOffPolicyBuffer): Vector off-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        try:
            for _ in track(
                range(rollout_step),
                description=f'Processing rollout for epoch: {logger.current_epoch}...',
            ):
                self._rollout_step(agent, buffer, logger, use_rand_action)
        except errors.LiveError:
            for _ in range(rollout_step):
                self._rollout_step(agent, buffer, logger, use_rand_action)

    def _rollout_step(  # pylint: disable=too-many-locals
        self,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        if use_rand_action:
            act = torch.as_tensor(self._env.sample_action(), dtype=torch.float32).to(
                self._device,
            )
        else:
            act = agent.step(self._current_obs, deterministic=False)

        next_obs, reward, cost, terminated, truncated, info = self.step(act)

        self._log_value(reward=reward, cost=cost, info=info)
        real_next_obs = next_obs.clone()
        for idx, done in enumerate(torch.logical_or(terminated, truncated)):
            if done:
                if 'final_observation' in info:
                    real_next_obs[idx] = info['final_observation'][idx]
                self._log_metrics(logger, idx)
                self._reset_log(idx)

        buffer.store(
            obs=self._current_obs,
            act=act,
            reward=reward,
            cost=cost,
            done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
            next_obs=real_next_obs,
        )

        self._current_obs = next_obs
