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

from typing import Any

import torch
from gymnasium.spaces import Box
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.base.pets import PETS
from omnisafe.algorithms.model_based.planner.arc import ARCPlanner
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.typing import OmnisafeSpace


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class LOOP(PETS):
    """The Learning Off-Policy with Online Planning (LOOP) algorithm.

    References:
        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `LOOP <https://arxiv.org/abs/2008.10066>`_
    """

    _log_alpha: torch.Tensor
    _alpha_optimizer: optim.Optimizer
    _target_entropy: float

    def _init_model(self) -> None:
        """Initialize the dynamics model and the planner.

        LOOP uses following models:

        - dynamics model: to predict the next state and the cost.
        - actor_critic: to predict the action and the value.
        - planner: to generate the action.
        """
        self._dynamics_state_space: OmnisafeSpace = (
            self._env.coordinate_observation_space
            if self._env.coordinate_observation_space is not None
            else self._env.observation_space
        )
        assert self._dynamics_state_space is not None and isinstance(
            self._dynamics_state_space.shape,
            tuple,
        )
        assert self._env.action_space is not None and isinstance(
            self._env.action_space.shape,
            tuple,
        )
        if isinstance(self._env.action_space, Box):
            self._action_space = self._env.action_space
        else:
            raise NotImplementedError
        self._actor_critic: ConstraintActorQCritic = ConstraintActorQCritic(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        self._use_actor_critic: bool = True
        self._update_count: int = 0
        self._dynamics: EnsembleDynamicsModel = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_shape=self._dynamics_state_space.shape,
            action_shape=self._env.action_space.shape,
            actor_critic=self._actor_critic,
            rew_func=None,
            cost_func=None,
            terminal_func=None,
        )
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)
        self._planner: ARCPlanner = ARCPlanner(
            dynamics=self._dynamics,
            planner_cfgs=self._cfgs.planner_cfgs,
            gamma=float(self._cfgs.algo_cfgs.gamma),
            cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device=self._device,
            actor_critic=self._actor_critic,
        )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        super()._init()

        self._alpha = self._cfgs.algo_cfgs.alpha
        self._alpha_gamma = self._cfgs.algo_cfgs.alpha_gamma
        self._policy_buf = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.algo_cfgs.policy_batch_size,
            device=self._device,
        )

    def _alpha_discount(self) -> None:
        """Alpha discount."""
        self._alpha *= self._alpha_gamma

    def _init_log(self) -> None:
        """Initialize logger.

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Value/alpha             | The value of alpha.                                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Values/reward_critic    | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        """
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

    def _save_model(self) -> None:
        """Save the model."""
        what_to_save: dict[str, Any] = {}
        # set up model saving
        what_to_save = {
            'dynamics': self._dynamics.ensemble_model,
            'actor_critic': self._actor_critic,
        }
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def _select_action(  # pylint: disable=unused-argument
        self,
        current_step: int,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Select action.

        Args:
            current_step (int): The current step.
            state (torch.Tensor): The current state.

        Returns:
            The selected action.
        """
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
        else:
            action, info = self._planner.output_action(state)
            self._logger.store(**info)

        assert action.shape == torch.Size(
            [1, *self._action_space.shape],
        ), 'action shape should be [batch_size, action_dim]'
        return action

    def _update_policy(self, current_step: int) -> None:
        """Update policy.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_policy_iters`` times.

        Args:
            current_step (int): The current step.
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
                    # freeze Q-network so you don't waste computational effort
                    # computing gradients for it during the policy learning step
                    for param in self._actor_critic.reward_critic.parameters():
                        param.requires_grad = False
                    if self._cfgs.algo_cfgs.use_cost:
                        for param in self._actor_critic.cost_critic.parameters():
                            param.requires_grad = False

                    self._update_actor(obs)

                    # unfreeze Q-network so you can optimize it at next DDPG step.
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
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_state: torch.Tensor,
        info: dict[str, Any],
    ) -> None:  # pylint: disable=too-many-arguments
        """Store real data in buffer.

        Args:
            state (torch.Tensor): The state from the environment.
            action (torch.Tensor): The action from the agent.
            reward (torch.Tensor): The reward signal from the environment.
            cost (torch.Tensor): The cost signal from the environment.
            terminated (torch.Tensor): The terminated signal from the environment.
            truncated (torch.Tensor): The truncated signal from the environment.
            next_state (torch.Tensor): The next state from the environment.
            info (dict[str, Any]): The information from the environment.
        """
        done = terminated or truncated
        goal_met = info.get('goal_met', False)
        if not done and not goal_met:
            # when goal_met == true:
            # current goal position is not related to the last goal position,
            # this huge transition will confuse the dynamics model.
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

        - Get the TD loss of reward critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
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
        """Update cost critic using TD3 algorithm.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
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

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
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
        r"""Computing ``pi/actor`` loss.

        The loss function in SAC is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \alpha \log \pi (s)

        where :math:`Q^V` is the min value of two reward critic networks, and :math:`\pi` is the
        policy network, and :math:`\alpha` is the temperature parameter.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(
            obs,
            deterministic=self._cfgs.algo_cfgs.loss_pi_deterministic,
        )
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()
