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
"""Robust Barrier Function Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.robust_barrier_solver import CBFQPLayer
from omnisafe.common.robust_gp_model import DynamicsModel
from omnisafe.envs.wrapper import CostNormalize, RewardNormalize, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config


class RobustBarrierFunctionAdapter(OffPolicyAdapter):
    """Robust Barrier Function Adapter for OmniSafe.

    :class:`RobustBarrierFunctionAdapter` is used to adapt the environment with RCBF controller.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`RobustBarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.solver: CBFQPLayer
        self.dynamics_model: DynamicsModel
        self._current_steps = 0
        self._num_episodes = 0

    def _wrapper(
        self,
        obs_normalize: bool = False,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. warning::
            Since solving the optimization problem requires obtaining physical quantities with
            practical significance from state observations, the Barrier Function Adapter does not
            support normalization of observations.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to False.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert not obs_normalize, 'Barrier function does not support observation normalization!'
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def set_solver(self, solver: CBFQPLayer) -> None:
        """Set the barrier function solver for Pendulum environment."""
        self.solver = solver
        self.solver.env = self._env  # type: ignore

    def set_dynamics_model(self, dynamics_model: DynamicsModel) -> None:
        """Set the dynamics model."""
        self.dynamics_model = dynamics_model
        self.dynamics_model.env = self._env  # type: ignore

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        assert self._eval_env
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
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
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
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            state = self.dynamics_model.get_state(self._current_obs)
            self._current_steps += 1
            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).unsqueeze(0).to(self._device)  # type: ignore
            else:
                act = agent.step(self._current_obs, deterministic=False)

            final_act = self.get_safe_action(obs=self._current_obs, act=act)

            next_obs, reward, cost, terminated, truncated, info = self.step(final_act)
            self._log_value(reward=reward, cost=cost, info=info)

            buffer.store(
                obs=self._current_obs,
                act=final_act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=next_obs,
            )

            if (
                self._ep_len[0] % 2 == 0
                and self._num_episodes < self._cfgs.dynamics_model_cfgs.gp_max_episodes
            ):
                next_state = self.dynamics_model.get_state(next_obs)
                self.dynamics_model.append_transition(
                    state.cpu().detach().numpy(),
                    final_act.cpu().detach().numpy(),
                    next_state.cpu().detach().numpy(),
                )

            self._current_obs = next_obs
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
                    self._num_episodes += 1
                    self._current_obs, _ = self._env.reset()

    @property
    def safe_action_space(self) -> OmnisafeSpace:
        """Return the action space in the safe domain."""
        if hasattr(self._env, 'safe_action_space'):
            return self._env.safe_action_space
        return self._env.action_space

    def get_safe_action(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Computes a safe action by applying robust barrier function.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            act (torch.Tensor): The proposed action to be evaluated for safety.

        Returns:
            torch.Tensor: The safe action to be executed in the environment.
        """
        state_batch = self.dynamics_model.get_state(obs)
        mean_pred_batch, sigma_pred_batch = self.dynamics_model.predict_disturbance(state_batch)

        return self.solver.get_safe_action(
            state_batch,
            act,
            mean_pred_batch,
            sigma_pred_batch,
        )

    def __getattr__(self, name: str) -> Any:
        """Return the unwrapped environment attributes."""
        return getattr(self._env, name)
