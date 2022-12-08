# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Safe controllers which do a black box optimization incorporating the constraint costs."""

import copy

import numpy as np
import scipy.stats as stats
import torch


class Planner:  # pylint: disable=too-many-instance-attributes
    """Model Predictive Control"""

    # pylint: disable-next=too-many-locals, too-many-arguments
    def __init__(
        self,
        device,
        env,
        models,
        actor_critic,
        horizon,
        reward_horizon,
        popsize,
        particles,
        max_iters,
        alpha,
        mixture_coefficient,
        kappa,
        safety_threshold,
        minimal_elites,
        num_elites,
    ):
        self.device = device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.env = copy.deepcopy(env)
        self.models = models
        self.actor_critic = actor_critic
        self.termination_function = default_termination_function
        self.horizon = horizon
        self.sol_dim = self.env.action_space.shape[0] * horizon
        self.action_max = np.repeat(self.env.action_space.high, self.horizon, axis=0)
        self.action_min = np.repeat(self.env.action_space.low, self.horizon, axis=0)
        self.mean = np.zeros((self.sol_dim,))
        # Shape: [ H * action_dim, 1 ]
        self.gaussian_traj = popsize
        self.mixture_coefficient = mixture_coefficient
        self.particles = particles
        self.max_iters = max_iters
        self.alpha_plan = alpha
        self.kappa = kappa
        self.reward_horizon = reward_horizon
        self.safety_threshold = safety_threshold
        self.minimal_elites = minimal_elites
        self.actor_traj = int(self.mixture_coefficient * self.gaussian_traj)
        self.num_elites = num_elites

    def planner_reset(self):
        """Reset planner when the episode end."""
        self.mean = np.zeros((self.sol_dim,))

    def get_action(
        self, curr_state, env=None
    ):  # pylint: disable=too-many-statements, too-many-locals
        """Select action when interact with environment."""
        # Set the reward of initial state to zero.
        actor_state = np.array(
            [np.concatenate(([0], curr_state.copy()), axis=0)] * (self.actor_traj)
        )
        # Shape: [actor_traj, reward_dim + state_dim]

        curr_state = np.array(
            [np.concatenate(([0], curr_state.copy()), axis=0)]
            * ((self.gaussian_traj + self.actor_traj) * self.particles)
        )
        # Shape: [(N + actor_traj) * particles, reward_dim + state_dim]

        curr_state = np.expand_dims(curr_state, axis=0)
        # Shape: [1, (N + actor_traj) * particles, reward_dim + state_dim]

        curr_state = np.repeat(curr_state, self.models.model.network_size, 0)
        # Shape: [numEnsemble, (N + actor_traj) * particles, reward_dim + state_dim]

        # initial mean and var of the sampling normal dist
        # shift the current array to the left, clear the used action
        self.mean[: -self.action_dim] = self.mean[self.action_dim :]
        # Shape: [ H * action_dim, 1 ]

        # fill the last position with the last second  action
        self.mean[-self.action_dim :] = self.mean[-2 * self.action_dim : -self.action_dim]
        mean = self.mean
        # Shape: [ H * action_dim, 1 ]

        var = np.tile(
            np.square(self.env.action_space.high[0] - self.env.action_space.low[0]) / 16,
            [self.sol_dim],
        )
        # Shape: [ H * action_dim, 1 ]

        # Add trajectories using actions suggested by actors
        actor_trajectories = np.zeros((self.actor_traj, self.sol_dim))
        # Shape: [actor_traj, H * action_dim]

        actor_state = torch.FloatTensor(actor_state).to(self.device)
        # Shape: [actor_traj, reward_dim + state_dim]

        actor_state_m = actor_state[0, :].reshape(1, -1)
        # Shape: [1, reward_dim + state_dim]

        actor_state_m2 = actor_state[1, :].reshape(1, -1)
        # Shape: [1, reward_dim + state_dim]

        for current_horizon in range(self.horizon):
            # Use deterministic policy to plan a action trajectory
            actor_actions_m = self.actor_critic.act_batch(
                actor_state_m.reshape(1, -1)[:, 1:], deterministic=True
            )
            # Shape: [1, action_dim]

            # Use dynamics model to plan
            actor_state_m = self.models.get_forward_prediction_random_ensemble_t(
                actor_state_m[:, 1:], actor_actions_m
            )
            # Shape: [1, reward_dim + state_dim]

            # Store a planning action to action buffer
            actor_trajectories[
                0, current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim
            ] = (actor_actions_m.detach().cpu().numpy())

            # Using Stochastic policy to plan a action trajectory
            actor_actions = self.actor_critic.act_batch(actor_state_m2[:, 1:])
            # Shape: [1, action_dim]

            # Use dynamics model to plan
            actor_state_m2 = self.models.get_forward_prediction_random_ensemble_t(
                actor_state_m2[:, 1:], actor_actions
            )
            # Shape: [1, reward_dim + state_dim]

            # Copy the planning action of stochastic actor (actor_traj-1) times, and store to action buffer
            actor_trajectories[
                1:, current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim
            ] = (actor_actions.detach().cpu().numpy())

        # Create gaussian distribution.
        # mean is the zero vector, var is Unit Matrix
        gaussian = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        current_iter = 0
        while current_iter < self.max_iters:
            lb_dist, ub_dist = mean - self.action_min, self.action_max - mean

            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )

            # Generate random normal gaussian variable and multiply by the var, then add to the mean
            action_traj = (
                gaussian.rvs(size=(self.gaussian_traj, self.sol_dim)) * np.sqrt(constrained_var)
                + mean
            ).astype(np.float32)
            # Shape: [ N , H * action_dim]

            # Combine the actor action with gaussian action
            action_traj = np.concatenate((action_traj, actor_trajectories), axis=0)
            # Shape: [ N + actor_trajectories, H * action_dim]

            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj, self.particles, axis=0)
            # Shape: [ particles, N + actor_trajectories, H * action_dim]

            # actions clipped between -1 and 1
            action_traj = np.clip(action_traj, -1, 1)
            # Shape: [ particles, N + actor_trajectories, H * action_dim]

            states = (
                torch.from_numpy(np.expand_dims(curr_state.copy(), axis=0)).float().to(self.device)
            )
            # Shape: [1, network_size, (N + actor_traj) * particles, reward_dim + state_dim]

            actions = np.repeat(
                np.expand_dims(action_traj, axis=0), self.models.model.network_size, axis=0
            )
            # Shape: [ network_size, particles, N + actor_trajectories, H * action_dim]

            actions = torch.FloatTensor(actions).to(self.device)
            # Shape: [ network_size, particles, N + actor_trajectories, H * action_dim]

            for current_horizon in range(self.horizon):
                states_h = states[current_horizon, :, :, 1:]
                # [ network_size, (N + actor_traj) * particles, reward_dim + state_dim]

                next_states = self.models.get_forward_prediction_t(
                    states_h,
                    actions[
                        :,
                        :,
                        current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim,
                    ],
                )
                # [ network_size, (N + actor_traj) * particles, reward_dim + state_dim]
                states = torch.cat((states, next_states.unsqueeze(0)), axis=0)
            # [ horizon + 1, network_size, (N + actor_traj) * particles, reward_dim + state_dim]

            states = states.cpu().detach().numpy()
            # [ horizon + 1, network_size, (N + actor_traj) * particles, reward_dim + state_dim]

            done = np.zeros((states.shape[1], states.shape[2], 1))
            # Shape :[network_size, (N + actor_traj) * particles, 1]

            # Set the reward of terminated states to zero
            for current_horizon in range(1, self.horizon + 1):
                for ens in range(states.shape[1]):
                    done[ens, :, :] = np.logical_or(
                        done[ens, :, :],
                        self.termination_function(None, None, states[current_horizon, ens, :, 1:]),
                    )
                    not_done = 1 - done[ens, :, :]
                    states[current_horizon, ens, :, 0] *= not_done.astype(np.float32).reshape(-1)

            # Find average cost of each trajectory
            returns = np.zeros((self.gaussian_traj + self.actor_traj,))
            safety_costs = np.zeros((self.gaussian_traj + self.actor_traj,))
            # Shape: [ N + actor_traj,  1 ]

            # This is the final action for evaluating terminated reward and cost
            final_action = (
                torch.from_numpy(
                    action_traj[
                        :,
                        (self.reward_horizon - 1)
                        * self.action_dim : (self.reward_horizon)
                        * self.action_dim,
                    ].reshape((self.gaussian_traj + self.actor_traj) * self.particles, -1)
                )
                .float()
                .to(self.device)
            )
            # Shape: [ (N + actor_traj) * particles,  action_dim ) ,
            # action_traj is [ (N + actor_traj) * particles, H * action_dim]

            final_action = final_action.repeat(self.models.model.network_size, 1)
            # Shape: [ (N + actor_traj) * particles, network_size , action_dim )

            # This is the final state for evaluating terminated reward and cost
            final_state = (
                torch.from_numpy(
                    states[self.reward_horizon - 1, :, :, 1:].reshape(
                        (self.gaussian_traj + self.actor_traj) * self.particles * states.shape[1],
                        -1,
                    )
                )
                .float()
                .to(self.device)
            )
            # [ (N + actor_traj) * particles, state_dim ]

            terminal_reward = self.actor_critic.q1(final_state, final_action).cpu().detach().numpy()
            terminal_reward = terminal_reward.reshape(states.shape[1], -1)
            # [ (N + actor_traj) * particles, 1]

            states_flatten = states[:, :, :, 1:].reshape(-1, self.obs_dim)
            # [ horizon * network_size * (N + actor_traj) * particles, state_dim]

            all_safety_costs = np.zeros((states_flatten.shape[0],))
            # [ horizon * network_size * (N + actor_traj) * particles, 1]

            all_safety_costs = env.get_observation_cost(states_flatten)
            # [ horizon * network_size * (N + actor_traj) * particles, 1]

            all_safety_costs = all_safety_costs.reshape(
                states.shape[0], states.shape[1], states.shape[2], 1
            )
            # [ horizon, network_size, (N + actor_traj) * particles, 1]

            # Calculate the average reward and max cost of N action trajectory,
            # each action trajectory have generated (network_size * particles) state-action trajectory
            # using (network_size * particles) ensemble models.
            for ensemble in self.models.model.elite_model_idxes:
                done[ensemble, :, :] = np.logical_or(
                    done[ensemble, :, :],
                    self.termination_function(
                        None, None, states[self.horizon - 1, ensemble, :, 1:]
                    ),
                )
                not_done = 1 - done[ensemble, :, :]
                q_rews = terminal_reward[ensemble, :] * not_done.reshape(-1)
                traj_indices = np.arange(0, self.gaussian_traj + self.actor_traj, 1).astype(int)
                for particle in range(self.particles):
                    returns[traj_indices] += (
                        np.sum(
                            states[
                                : self.reward_horizon,
                                ensemble,
                                traj_indices * self.particles + particle,
                                0,
                            ],
                            axis=0,
                        )
                        + q_rews.reshape(-1)[traj_indices * self.particles + particle]
                    )
                    safety_costs[traj_indices] = np.maximum(
                        safety_costs,
                        np.sum(
                            all_safety_costs[
                                0 : self.horizon,
                                ensemble,
                                traj_indices * self.particles + particle,
                                0,
                            ],
                            axis=0,
                        ),
                    )

            returns /= states.shape[1] * self.particles
            # [ N + actor_traj, 1]
            costs = -returns
            # [ N + actor_traj, 1]

            if (safety_costs < self.safety_threshold).sum() < self.minimal_elites:
                safety_rewards = -safety_costs
                # [ N + actor_traj, 1]

                max_safety_reward = np.max(safety_rewards)
                # [1,1]

                score = np.exp(self.kappa * (safety_rewards - max_safety_reward))
                # [ N + actor_traj, 1]

                indices = np.argsort(safety_costs)
                # [ N + actor_traj, 1]

                mean = np.sum(
                    action_traj[
                        np.arange(0, self.gaussian_traj + self.actor_traj, 1).astype(int)
                        * self.particles,
                        :,
                    ]
                    * score.reshape(-1, 1),
                    axis=0,
                ) / (np.sum(score) + 1e-10)
                # [1, H * action_dim],  action_traj is [ N + actor_traj, H * action_dim], score is  [ N + actor_traj, 1]

                new_var = np.average(
                    (
                        action_traj[
                            np.arange(0, self.gaussian_traj + self.actor_traj, 1).astype(int)
                            * self.particles,
                            :,
                        ]
                        - mean
                    )
                    ** 2,
                    weights=score.reshape(-1),
                    axis=0,
                )
            # [ 1,  H * action_dim]

            else:  # if have enough safe trajectory
                # safe trajectory's costs is -reward, unsafe trajectory's costs is 1e4
                costs = (safety_costs < self.safety_threshold) * costs + (
                    safety_costs >= self.safety_threshold
                ) * 1e4
                # [ N + actor_traj, 1]

                # select indices of safe trajectory
                indices = np.arange(costs.shape[0])
                indices = np.array([idx for idx in indices if costs[idx] < 1e3])
                # [ N + actor_traj, 1]

                if indices.shape[0] == 0 or action_traj.shape[0] == 0:
                    break
                safe_action_traj = action_traj[
                    np.arange(0, self.gaussian_traj + self.actor_traj, 1).astype(int)
                    * self.particles,
                    :,
                ][indices, :]
                # [ num_safe_traj, H * action_dim]

                # use safe trajectory and its reward as weight to update
                rewards = -costs[indices]
                # [ num_safe_traj, 1 ]

                max_reward = np.max(rewards)
                # [ 1, 1 ]

                score = np.exp(self.kappa * (rewards - max_reward))
                # [ num_safe_traj, 1 ]

                mean = np.sum(safe_action_traj * score.reshape(-1, 1), axis=0) / (
                    np.sum(score) + 1e-10
                )
                # [1, H * action_dim] = [1, H * action_dim] / [1,1]

                new_var = np.average(
                    (safe_action_traj - mean) ** 2, weights=score.reshape(-1), axis=0
                )
            # [ 1,  H * action_dim]

            var = (self.alpha_plan) * var + (1 - self.alpha_plan) * new_var
            current_iter += 1

            # Initialize the var every 6 times
            if (current_iter + 1) % 6 == 0:
                var = np.tile(
                    np.square(self.env.action_space.high[0] - self.env.action_space.low[0]) / 16.0,
                    [self.sol_dim],
                ) * (1.5 ** ((current_iter + 1) // 6))

            # If safe trajectory not enough and t>5  or t>25 ,then break
            if (
                ((safety_costs < self.safety_threshold).sum() >= self.minimal_elites)
                and current_iter > 5
            ) or current_iter > 25:
                break

        # Store the mean and use it in next plan
        self.mean = mean

        # Return [1, action_dim], that is the first action of H horizon action mean, which shape is [1, H * action_dim]
        return mean[: self.action_dim]


def default_termination_function(state, action, next_state):  # pylint: disable=unused-argument
    '# Default termination function that outputs done=False'
    if torch.is_tensor(next_state):
        done = torch.zeros((next_state.shape[0], 1))
    else:
        done = np.zeros((next_state.shape[0], 1))
    return done
