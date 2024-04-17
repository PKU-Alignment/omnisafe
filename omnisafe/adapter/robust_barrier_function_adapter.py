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
import numpy as np

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.utils.config import Config
from omnisafe.common.robust_barrier_solver import CBFQPLayer
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.common.robust_gp_model import DynamicsModel


from omnisafe.envs.wrapper import (
    CostNormalize,
    RewardNormalize,
    Unsqueeze,
)

class RobustBarrierFunctionAdapter(OffPolicyAdapter):

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.solver = None
        self.compensator = None
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
            Since solving the optimization problem requires obtaining physical quantities with practical 
            significance from state observations, the Barrier Function Adapter does not support 
            normalization of observations.

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
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)
        # self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        # self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        
    def set_solver(self, solver: CBFQPLayer):
        """Set the barrier function solver for Pendulum environment."""
        self.solver: CBFQPLayer = solver
        self.solver.env = self._env

    def set_dynamics_model(self, dynamics_model: DynamicsModel):
        """Set the dynamics model."""
        self.dynamics_model = dynamics_model
        self.dynamics_model.env = self._env

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
            state = self.dynamics_model.get_state(self._current_obs) # 动态模型将观测转换为状态，状态和观测之间有一个互逆的转换
            self._current_steps += 1
            if use_rand_action:
                act = torch.normal(torch.zeros(self.action_space.shape), torch.ones(self.action_space.shape)).unsqueeze(0).to(self._device)
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
            
            if self._ep_len[0] % 2 == 0 and self._num_episodes < self._cfgs.dynamics_model_cfgs.gp_max_episodes:
                next_state = self.dynamics_model.get_state(next_obs)
                self.dynamics_model.append_transition(state.cpu().detach().numpy(), final_act.cpu().detach().numpy(), next_state.cpu().detach().numpy(), t_batch=np.array([self._ep_len[0]*self._env.dt]))
                
            self._current_obs = next_obs
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
                    self._num_episodes += 1
                    self._current_obs, _ = self._env.reset()
            
    @property
    def safe_action_space(self) -> OmnisafeSpace:
        if hasattr(self._env, 'safe_action_space'):
            return self._env.safe_action_space
        else:
            return self._env.action_space
            
    def get_safe_action(self, obs, act, modular=False, cbf_info_batch=None):
        """Given a nominal action, returns a minimally-altered safe action to take.

        Parameters
        ----------
        obs : torch.tensor
        act : torch.tensor
        dynamics_model : DynamicsModel

        Returns
        -------
        safe_act : torch.tensor
            Safe actions to be taken (cbf_action + action).
        """
        state_batch = self.dynamics_model.get_state(obs)
        mean_pred_batch, sigma_pred_batch = self.dynamics_model.predict_disturbance(state_batch)
        safe_act = self.solver.get_safe_action(state_batch, act, mean_pred_batch, sigma_pred_batch, modular=modular, cbf_info_batch=cbf_info_batch)

        return safe_act

    def __getattr__(self, name):
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")