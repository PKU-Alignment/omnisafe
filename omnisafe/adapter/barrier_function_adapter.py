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
"""BarrierFunction Adapter for OmniSafe."""

from __future__ import annotations

import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.common.barrier_comp import BarrierCompensator

from omnisafe.envs.wrapper import (
    AutoReset,
    CostNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)

class BarrierFunctionAdapter(OnPolicyAdapter):
    """BarrierFunction Adapter for OmniSafe.

    The BarrierFunction Adapter is used to establish the logic of interaction between agents and the 
    environment based on control barrier functions. Its key feature is the introduction of action 
    compensators and barrier function solvers.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.solver = None
        self.compensator = None
        self.first_iter = 1

    def _wrapper(
        self,
        obs_normalize: bool = False,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.
        
        .. warning::
            Since solving the optimization problem requires obtaining physical quantities with practical 
            significance from state observations, the Barrier Function Adapter does not support 
            normalization of observations.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to False.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert not obs_normalize, 'Barrier function does not support observation normalization!'
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, time_limit=1000, device=self._device)
            self._eval_env = TimeLimit(self._eval_env, time_limit=1000, device=self._device)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    def set_solver(self, solver: PendulumSolver):
        """Set the barrier function solver for Pendulum environment."""
        self.solver: PendulumSolver = solver
        
    def set_compensator(self, compensator: BarrierCompensator):
        """Set the action compensator."""
        self.compensator: BarrierCompensator = compensator

    def reset_gp_model(self):
        """Reset the gaussian processing model of barrier function solver."""
        self.solver.GP_model_prev = self.solver.GP_model.copy()
        self.solver.build_GP_model()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()
        if not self.first_iter:
            self.reset_gp_model()

        obs, _ = self.reset()
        while abs(self._env.unwrapped.state[0]) > 1:
            obs, _ = self._env.reset()
        path_obs = []
        path_act = []
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            with torch.no_grad():
                value_r = agent.reward_critic(obs)[0]
                value_c = agent.cost_critic(obs)[0]
                act_dist = agent.actor(obs)
                act_mean, act_std = act_dist.mean, agent.actor.std

                approx_compensating_act = self.compensator(obs=obs)
                compensated_act_mean_raw = act_mean + approx_compensating_act
                
                if self.first_iter:
                    [f, g, x, std] = self.solver.get_GP_dynamics(obs, use_prev_model = False)
                else:
                    [f, g, x, std] = self.solver.get_GP_dynamics(obs, use_prev_model = True)
                
                compensating_act = self.solver.control_barrier(compensated_act_mean_raw, f, g, x, std)

                compensated_act_mean = compensated_act_mean_raw + compensating_act
                final_act = torch.normal(compensated_act_mean, act_std)
            
            logp = agent.actor.log_prob(final_act).detach()
            path_obs.append(obs.detach().cpu().squeeze().numpy())
            path_act.append(final_act.detach().cpu().squeeze().numpy())

            next_obs, reward, cost, terminated, truncated, info = self.step(final_act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})
            logger.store({'Metrics/angle': cost})

            buffer.store(
                obs=obs,
                act=final_act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
                approx_compensating_act=approx_compensating_act.detach(),
                compensating_act=compensating_act.detach(),
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                obs[idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                        if step < 650:
                            self.solver.update_GP_dynamics(obs = path_obs, act = path_act)

                        path_obs = []
                        path_act = []
                        obs, _ = self.reset()
                        while abs(self._env.unwrapped.state[0]) > 1:
                            obs, _ = self._env.reset()
                    buffer.finish_path(last_value_r, last_value_c, idx)
        self.first_iter = 0

