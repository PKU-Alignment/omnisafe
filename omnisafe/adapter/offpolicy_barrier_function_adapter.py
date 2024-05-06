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
"""BarrierFunction OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.gaussian_process import GaussianProcessRegressor

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import CostNormalize, RewardNormalize, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class OffPolicyBarrierFunctionAdapter(OffPolicyAdapter):
    """OffPolicy Barrier Function Adapter for OmniSafe.

    :class:`OffPolicyBarrierFunctionAdapter` is used to adapt the environment with CBF controller.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.solver: PendulumSolver
        self.compensator: BarrierCompensator
        self.first_iter: int = 1
        self.episode_rollout: dict[str, Any] = {}
        self.episode_rollout['obs'] = []
        self.episode_rollout['final_act'] = []
        self.episode_rollout['approx_compensating_act'] = []
        self.episode_rollout['compensating_act'] = []

    def _wrapper(
        self,
        obs_normalize: bool = False,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        assert not obs_normalize, 'Barrier function does not support observation normalization!'
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment in an evaluation environment.

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
                final_act = self.get_safe_action(obs=obs, act=act, is_eval=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(final_act)
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

    def set_solver(self, solver: PendulumSolver) -> None:
        """Set the barrier function solver for Pendulum environment."""
        self.solver = solver

    def set_compensator(self, compensator: BarrierCompensator) -> None:
        """Set the action compensator."""
        self.compensator = compensator

    def reset_gp_model(self) -> None:
        """Reset the gaussian processing model of barrier function solver."""
        self.solver.reset_gp_model()

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout in off-policy manner with barrier function controller.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).unsqueeze(0).to(self._device)  # type: ignore
            else:
                act = agent.actor.predict(self._current_obs, deterministic=False)

            final_act = self.get_safe_action(self._current_obs, act)

            self.episode_rollout['obs'].append(self._current_obs)
            self.episode_rollout['final_act'].append(final_act)

            next_obs, reward, cost, terminated, truncated, info = self.step(final_act)
            self._log_value(reward=reward, cost=cost, info=info)

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=next_obs,
            )

            self._current_obs = next_obs
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    self._log_metrics(logger, idx)
                    compensator_loss = self.compensator.update(
                        torch.cat(self.episode_rollout['obs']),
                        torch.cat(self.episode_rollout['approx_compensating_act']),
                        torch.cat(self.episode_rollout['compensating_act']),
                    )
                    logger.store({'Value/Loss_compensator': compensator_loss.item()})
                    self.solver.update_gp_dynamics(
                        obs=torch.cat(self.episode_rollout['obs']),  # type: ignore
                        act=torch.cat(self.episode_rollout['final_act']),  # type: ignore
                    )

                    self.episode_rollout['obs'] = []
                    self.episode_rollout['final_act'] = []
                    self.episode_rollout['approx_compensating_act'] = []
                    self.episode_rollout['compensating_act'] = []

                    self._reset_log(idx)
                    self._current_obs, _ = self._env.reset()
                    self.first_iter = 0
                    if not self.first_iter:
                        self.reset_gp_model()

    def get_safe_action(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        is_eval: bool = False,
    ) -> torch.Tensor:
        """Computes a safe action by applying compensatory actions.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            act (torch.Tensor): The proposed action to be evaluated for safety.
            is_eval (bool, optional): A flag to indicate whether this is an evaluation phase, defaulting to False.

        Returns:
            torch.Tensor: The safe action to be executed in the environment.
        """
        with torch.no_grad():
            approx_compensating_act = self.compensator(obs=self._current_obs)
            compensated_act_mean_raw = act + approx_compensating_act

            if self.first_iter:
                [f, g, x, std] = self.solver.get_gp_dynamics(obs, use_prev_model=False)
            else:
                [f, g, x, std] = self.solver.get_gp_dynamics(obs, use_prev_model=True)

            compensating_act = self.solver.control_barrier(compensated_act_mean_raw, f, g, x, std)
            safe_act = compensated_act_mean_raw + compensating_act

            if not is_eval:
                self.episode_rollout['compensating_act'].append(compensating_act)
                self.episode_rollout['approx_compensating_act'].append(approx_compensating_act)

        return safe_act

    @property
    def gp_models(self) -> list[GaussianProcessRegressor]:
        """Return the gp models to be saved."""
        return self.solver.gp_models
