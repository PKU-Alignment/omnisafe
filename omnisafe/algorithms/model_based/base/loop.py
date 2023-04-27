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
"""Implementation of the Learning Off-Policy with Online Planning algorithm."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import ModelBasedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.base.pets import PETS
from omnisafe.algorithms.model_based.planner.arc import ARCPlanner
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class LOOP(PETS):
    """The Learning Off-Policy with Online Planning (LOOP) algorithm.

    References:

        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `LOOP <https://arxiv.org/abs/2008.10066>`_
    """

    def _init_model(self) -> None:
        if self._env.action_space is not None and len(self._env.action_space.shape) > 0:
            self._action_dim = self._env.action_space.shape[0]
        else:
            # error handling for action dimension is none of shape of action less than 0
            raise ValueError('Action dimension is None or less than 0')
        if self._env.action_space is not None:
            self._action_dim = self._env.action_space.shape[0]

        self._dynamics_state_space = (
            self._env.coordinate_observation_space
            if self._env.coordinate_observation_space is not None
            else self._env.observation_space
        )

        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        self._use_actor_critic = True
        self._update_count = 0
        self._dynamics = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_shape=self._dynamics_state_space.shape,
            action_shape=self._env.action_space.shape,
            reward_size=1,
            cost_size=1,
            use_cost=False,
            use_terminal=False,
            use_var=False,
            use_reward_critic=True,
            use_cost_critic=False,
            actor_critic=self._actor_critic,
            rew_func=None,
            cost_func=None,
            terminal_func=None,
        )
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)
        self._planner = ARCPlanner(
            dynamics=self._dynamics,
            actor_critic=self._actor_critic,
            num_models=self._cfgs.dynamics_cfgs.num_ensemble,
            horizon=self._cfgs.algo_cfgs.plan_horizon,
            num_iterations=self._cfgs.algo_cfgs.num_iterations,
            num_particles=self._cfgs.algo_cfgs.num_particles,
            num_samples=self._cfgs.algo_cfgs.num_samples,
            num_elites=self._cfgs.algo_cfgs.num_elites,
            mixture_coefficient=self._cfgs.algo_cfgs.mixture_coefficient,
            temperature=self._cfgs.algo_cfgs.temperature,
            momentum=self._cfgs.algo_cfgs.momentum,
            epsilon=self._cfgs.algo_cfgs.epsilon,
            init_var=self._cfgs.algo_cfgs.init_var,
            gamma=self._cfgs.algo_cfgs.gamma,
            device=self._device,
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._env.action_space.shape,
            action_max=1.0,
            action_min=-1.0,
        )

    def _init(self) -> None:
        super()._init()
        self._log_alpha: torch.Tensor
        self._alpha_optimizer: optim.Optimizer
        self._target_entropy: float

        self._alpha = self._cfgs.algo_cfgs.alpha
        self._alpha_gamma = self._cfgs.algo_cfgs.alpha_gamma
        self._policy_buf = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.algo_cfgs.policy_batch_size,
            device=self._device,
        )

    def _alpha_discount(self):
        """Alpha discount."""
        self._alpha *= self._alpha_gamma

    def _init_log(self) -> None:
        """Initialize logger."""
        super()._init_log()
        self._logger.register_key('Value/alpha')
        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

    def _select_action(  # pylint: disable=unused-argument
        self,
        current_step: int,
        state: torch.Tensor,
        env: ModelBasedAdapter,
    ) -> tuple[torch.Tensor, dict]:
        """Select action.

        Args:
            current_step (int): current step
            state (torch.Tensor): current state
            env (ModelBasedAdapter): environment

        Returns:
            action (torch.Tensor): action
            action_info (dict): action information
        """
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
        else:
            action, info = self._planner.output_action(state)
            # action = self._actor_critic.actor.predict(state, deterministic=False).detach()
            self._logger.store(**info)

        assert action.shape == torch.Size(
            [state.shape[0], self._action_dim],
        ), 'action shape should be [batch_size, action_dim]'

        info = {}
        return action, info

    def _update_policy(self, current_step: int) -> None:
        """Update policy.

        Args:
            current_step (int): current step
        """
        if current_step >= self._cfgs.algo_cfgs.start_learning_steps:
            for _step in range(self._cfgs.algo_cfgs.update_policy_iters):
                self._update_count += 1

                data = self._policy_buf.sample_batch()
                obs, act, reward, cost, done, next_obs = (
                    data['obs'],
                    data['act'],
                    data['reward'],
                    data['cost'],
                    data['done'],
                    data['next_obs'],
                )

                self._update_reward_critic(obs, act, reward, done, next_obs)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, act, cost, done, next_obs)

                if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                    # Freeze Q-network so you don't waste computational effort
                    # computing gradients for it during the policy learning step.
                    for param in self._actor_critic.reward_critic.parameters():
                        param.requires_grad = False
                    if self._cfgs.algo_cfgs.use_cost:
                        for param in self._actor_critic.cost_critic.parameters():
                            param.requires_grad = False

                    self._update_actor(obs)

                    # Unfreeze Q-network so you can optimize it at next DDPG step.
                    for param in self._actor_critic.reward_critic.parameters():
                        param.requires_grad = True
                    if self._cfgs.algo_cfgs.use_cost:
                        for param in self._actor_critic.cost_critic.parameters():
                            param.requires_grad = True

                    self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

                if self._cfgs.algo_cfgs.alpha_discount:
                    self._alpha_discount()

    def _store_real_data(  # pylint: disable=too-many-arguments,unused-argument
        self,
        current_step: int,
        ep_len: int,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_state: torch.Tensor,
        info: dict,
        action_info: dict,
    ) -> None:  # pylint: disable=too-many-arguments
        """Store real data in buffer.

        Args:
            current_step (int): current step
            ep_len (int): episode length
            state (torch.Tensor): current state
            action (torch.Tensor): action
            reward (torch.Tensor): reward
            cost (torch.Tensor): cost
            terminated (torch.Tensor): terminated
            truncated (torch.Tensor): truncated
            next_state (torch.Tensor): next state
            info (dict): information
            action_info (dict): action information
        """
        done = terminated or truncated
        goal_met = False if 'goal_met' not in info.keys() else info['goal_met']
        if not done and not goal_met:
            # pylint: disable-next=line-too-long
            # if goal_met == true, Current goal position is not related to the last goal position, this huge transition will confuse the dynamics model.
            self._dynamics_buf.store(
                obs=state,
                act=action,
                reward=reward,
                cost=cost,
                next_obs=next_state,
                done=done,
            )
        if (done and self._cfgs.algo_cfgs.policy_store_done) or (not done and not goal_met):
            self._policy_buf.store(
                obs=state,
                act=action,
                reward=reward,
                cost=cost,
                next_obs=next_state,
                done=done,
            )

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic using Soft Actor-Critic.

        Args:
            obs (torch.Tensor): observation
            action (torch.Tensor): action
            reward (torch.Tensor): reward
            done (torch.Tensor): done
            next_obs (torch.Tensor): next observation
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()

        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, target_q_value_r) + nn.functional.mse_loss(
            q2_value_r,
            target_q_value_r,
        )
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        loss.backward()

        if self._cfgs.algo_cfgs.use_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update cost critic using TD3 algorithm.

        Args:
            obs (torch.Tensor): current observation
            act (torch.Tensor): current action
            cost (torch.Tensor): current cost
            done (torch.Tensor): current done signal
            next_obs (torch.Tensor): next observation

        Returns:
            None
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)[0]
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.use_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor using Soft Actor-Critic algorithm.

        Args:
            obs (torch.Tensor): observation
        """
        self._actor_critic.actor_optimizer.zero_grad()
        loss = self._loss_pi(obs)
        loss.backward()
        if self._cfgs.algo_cfgs.use_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

        self._logger.store(
            **{
                'Value/alpha': self._alpha,
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for actor using Soft Actor-Critic algorithm.

        Args:
            obs (torch.Tensor): observation
        """
        action = self._actor_critic.actor.predict(
            obs,
            deterministic=self._cfgs.algo_cfgs.loss_pi_deterministic,
        )
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()

    def _log_when_not_update(self) -> None:
        """Log when not update."""
        self._logger.store(
            **{
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                **{
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
