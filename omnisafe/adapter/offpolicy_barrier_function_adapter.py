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
"""OffPolicy Barrier Function Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.gaussian_process import GaussianProcessRegressor

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.gp_model import DynamicsModel
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import CostNormalize, RewardNormalize, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class OffPolicyBarrierFunctionAdapter(OffPolicyAdapter):
    """OffPolicy Barrier Function Adapter for OmniSafe.

    :class:`OffPolicyBarrierFunctionAdapter` is used to adapt the environment with a CBF controller,
    mapping the agent actions from unsafe ones to safe ones.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.

    Attributes:
        solver (PendulumSolver): The solver used for the environment, currently supporting
                                ``Pendulum-v1``.
        dynamics_model (DynamicsModel): The dynamics model used to predict the environment's behavior.
        compensator (BarrierCompensator): The compensator used to approximate previous actions.
        first_iter (bool): A flag indicating if it is the first iteration.
        episode_rollout (dict[str, Any]): A dictionary to store the episode rollout information,
                                          including observations and various actions,
                                          useful for updating compensator.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`OffPolicyBarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)

        if env_id == 'Pendulum-v1':
            self.solver: PendulumSolver = PendulumSolver(
                action_size=self.action_space.shape[0],  # type: ignore
                device=self._device,
            )
            self.dynamics_model: DynamicsModel = DynamicsModel(
                observation_size=self.observation_space.shape[0],  # type: ignore
            )
        else:
            raise NotImplementedError(f'Please implement solver for {env_id} !')
        self.compensator: BarrierCompensator = BarrierCompensator(
            obs_dim=self.observation_space.shape[0],  # type: ignore
            act_dim=self.action_space.shape[0],  # type: ignore
            cfgs=cfgs.compensator_cfgs,
        ).to(self._device)

        self.first_iter: bool = True
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

    def reset_gp_model(self) -> None:
        """Reset the gaussian processing model of barrier function solver."""
        self.dynamics_model.reset_gp_model()

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout in off-policy manner with the ``dynamics_model``, ``solver`` and ``compensator``.

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
                    self.dynamics_model.update_gp_dynamics(
                        obs=torch.cat(self.episode_rollout['obs']),  # type: ignore
                        act=torch.cat(self.episode_rollout['final_act']),  # type: ignore
                    )

                    self.episode_rollout['obs'] = []
                    self.episode_rollout['final_act'] = []
                    self.episode_rollout['approx_compensating_act'] = []
                    self.episode_rollout['compensating_act'] = []

                    self._reset_log(idx)
                    self._current_obs, _ = self._env.reset()
                    self.first_iter = False
                    self.reset_gp_model()

    def get_safe_action(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        is_eval: bool = False,
    ) -> torch.Tensor:
        """Computes a safe action by applying compensatory actions.

        .. note::
            This is the core method of the CBF method. Users can modify this function to implement
            customized action mapping.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            act (torch.Tensor): The proposed action to be controlled for safety.
            is_eval (bool, optional): A flag to indicate whether this is an evaluation phase, defaulting to False.

        Returns:
            torch.Tensor: The safe action to be executed in the environment.
        """
        with torch.no_grad():
            approx_compensating_act = self.compensator(obs=obs)
            compensated_act_mean_raw = act + approx_compensating_act

            [f, g, x, std] = self.dynamics_model.get_gp_dynamics(
                obs,
                use_prev_model=not self.first_iter,
            )
            compensating_act = self.solver.control_barrier(
                original_action=compensated_act_mean_raw,
                f=f,
                g=g,
                x=x,
                std=std,
            )
            safe_act = compensated_act_mean_raw + compensating_act

            if not is_eval:
                self.episode_rollout['compensating_act'].append(compensating_act)
                self.episode_rollout['approx_compensating_act'].append(approx_compensating_act)

        return safe_act

    @property
    def gp_models(self) -> list[GaussianProcessRegressor]:
        """Return the gp models to be saved."""
        return self.dynamics_model.gp_models
