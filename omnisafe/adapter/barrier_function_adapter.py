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
"""Barrier Function Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track
from sklearn.gaussian_process import GaussianProcessRegressor

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.gp_model import DynamicsModel
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import AutoReset, CostNormalize, RewardNormalize, TimeLimit, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class BarrierFunctionAdapter(OnPolicyAdapter):
    """Barrier Function Adapter for OmniSafe.

    The Barrier Function Adapter is used to establish the logic of interaction between agents and
    the environment based on control barrier functions. Its key feature is the introduction of
    action compensators and barrier function solvers.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BarrierFunctionAdapter`."""
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
        if self._env.need_time_limit_wrapper:
            assert (
                self._env.max_episode_steps
            ), 'You must define max_episode_steps as an integer\
                \nor cancel the use of the time_limit wrapper.'
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def reset_gp_model(self) -> None:
        """Reset the gaussian processing model of barrier function solver."""
        self.dynamics_model.reset_gp_model()

    def rollout(  # pylint: disable=too-many-locals,too-many-branches
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment with barrier function controller.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()

        obs, _ = self.reset()
        self.episode_rollout['obs'] = []
        self.episode_rollout['final_act'] = []
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            with torch.no_grad():
                value_r = agent.reward_critic(obs)[0]
                value_c = agent.cost_critic(obs)[0]
                act_dist = agent.actor(obs)
                act_mean, act_std = act_dist.mean, agent.actor.std

                safe_act = self.get_safe_action(
                    obs,
                    act_mean,
                    act_std,
                )
                logp = agent.actor.log_prob(safe_act)

            self.episode_rollout['obs'].append(obs)
            self.episode_rollout['final_act'].append(safe_act)

            next_obs, reward, cost, terminated, truncated, info = self.step(safe_act)
            self._log_value(reward=reward, cost=cost, info=info)

            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=safe_act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch

            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                obs[idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
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
                        obs, _ = self.reset()
                    buffer.finish_path(last_value_r, last_value_c, idx)
        self.first_iter = False
        self.reset_gp_model()

    def get_safe_action(
        self,
        obs: torch.Tensor,
        act_mean: torch.Tensor,
        act_std: torch.Tensor,
    ) -> torch.Tensor:
        """Computes a safe action by applying compensatory actions.

        .. note::
            This is the core method of the CBF method. Users can modify this function to implement
            customized action mapping.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            act_mean (torch.Tensor): The mean of proposed action to be controlled for safety.
            act_std (torch.Tensor): The standard deviation of proposed action to be controlled for safety.

        Returns:
            list(torch.Tensor): The safe actions for interaction and compensating actions for compensator training.
        """
        with torch.no_grad():
            approx_compensating_act = self.compensator(obs=obs)
            compensated_act_mean_raw = act_mean + approx_compensating_act

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

            compensated_act_mean = compensated_act_mean_raw + compensating_act
            safe_act = torch.normal(compensated_act_mean, act_std)
            self.episode_rollout['compensating_act'].append(compensating_act)
            self.episode_rollout['approx_compensating_act'].append(approx_compensating_act)

        return safe_act

    @property
    def gp_models(self) -> list[GaussianProcessRegressor]:
        """Return the gp models to be saved."""
        return self.dynamics_model.gp_models
