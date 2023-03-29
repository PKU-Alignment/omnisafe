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
"""Virtual Environment"""

import numpy as np
import torch


class VirtualEnv:
    """Virtual environment for generating data or planning"""

    def __init__(self, algo, model, env_name, device=torch.device('cpu')):
        self.algo = algo
        self.model = model
        self.env_name = env_name
        self.device = device
        if self.model.env_type == 'gym' and self.algo in ['MBPPOLag']:
            self.state_start_dim = 0
        elif self.model.env_type == 'gym' and self.algo in ['CAP', 'SafeLOOP']:
            self.state_start_dim = 1
        elif self.model.env_type == 'mujoco-velocity' and self.algo in [
            'MBPPOLag',
            'CAP',
            'SafeLOOP',
        ]:
            self.state_start_dim = 2

    def _termination_fn(self, env_name, obs, act, next_obs):
        """Terminal function"""
        if env_name == 'Hopper-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (
                np.isfinite(next_obs).all(axis=-1)
                * np.abs(next_obs[:, 1:] < 100).all(axis=-1)
                * (height > 0.7)
                * (np.abs(angle) < 0.2)
            )

            done = ~not_done
            done = done[:, None]
            return done
        if env_name == 'Walker2d-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        if 'walker_' in env_name:
            torso_height = next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.0
            else:
                offset = 0.26
            not_done = (
                (torso_height > 0.8 - offset)
                * (torso_height < 2.0 - offset)
                * (torso_ang > -1.0)
                * (torso_ang < 1.0)
            )
            done = not not_done
            done = done[:, None]
            return done

        return False

    def _get_logprob(self, input_data, means, variances):

        k = input_data.shape[-1]
        log_prob = (
            -1
            / 2
            * (
                k * np.log(2 * np.pi)
                + np.log(variances).sum(-1)
                + (np.power(input_data - means, 2) / variances).sum(-1)
            )
        )

        # [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        # [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds
    def vitual_step_ensemble_output(self, states, actions, rew_func=None, cost_func=None, truncated_func=None, idx=None, deterministic=False):
        # state, actions shape is [network_size, batch_size, state_dim/action_dim]
        # output is [network_size, batch_size, state_dim/action_dim]
        assert states.shape[:-1] == actions.shape[:-1], 'states and actions must have same shape'
        #assert states.shape[0] == self.network_size, 'states must have same shape with network size'
        #assert actions.shape[0] == self.network_size, 'actions must have same shape with network size'
        inputs = torch.cat((states, actions), dim=-1)
        ensemble_model_means, ensemble_model_vars = self.predict_t(inputs)
        ensemble_model_means[:, :, self.state_start_dim :] += states

        return ensemble_model_means, ensemble_model_vars

    def vitual_step_single_output(self, states, actions, rew_func=None, cost_func=None, truncated_func=None, idx=None, deterministic=False):
        # state, actions shape is [network_size, batch_size, state_dim/action_dim]
        # output is [1, batch_size, dim]
        assert states.shape[:-1] == actions.shape[:-1], 'states and actions must have same shape'
        #assert states.shape[0] == self.network_size, 'states must have same shape with network size'
        #assert actions.shape[0] == self.network_size, 'actions must have same shape with network size'
        inputs = torch.cat((states, actions), dim=-1)
        ensemble_model_means, ensemble_model_vars = self.predict_t(inputs)
        ensemble_model_means[:, :, self.state_start_dim :] += states

        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + torch.normal(size=ensemble_model_means.shape, device=self.device) * ensemble_model_stds
            )
        samples_result ={}

        if idx == None:
            _, batch_size, _ = ensemble_model_means.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)
            samples = ensemble_samples[model_idxes, batch_idxes]
            samples_mean = ensemble_model_means[model_idxes, batch_idxes]
            samples_var = ensemble_model_vars[model_idxes, batch_idxes]
        else:
            samples = ensemble_samples[idx]
            samples_mean = ensemble_model_means[idx]
            samples_var = ensemble_model_vars[idx]
        samples_result['state'] = samples
        if rew_func is not None:
            samples_result['reward'] = rew_func(states, actions, samples)
        if self.use_cost and cost_func is not None:
            samples_result['cost'] = cost_func(states, actions, samples)
        if truncated_func is not None:
            samples_result['truncated'] = truncated_func(states, actions, samples)
        samples_result['samples_mean'] = samples_mean
        samples_result['samples_var'] = samples_var


        return samples_result

    traj = self._dynamics.imagine(states_repeat, actions_repeat, self._horizon)
    # {states, rewards, values}, each value shape is [num_particles * num_samples, horizon]

    def imagine(self, states, horizon, return_ensemble_result=True, actions=None, actor_critic=None):
        # input states shapes is [batch_size, state_dim]
        # output shapes is [horizon, batch_size, dim]
        if actions is not None:
            assert actions.shape[0] == horizon, "actions must be equal to horizon"
            use_actor = False
        else:
            assert actor_critic is not None, "actor_critic must not be None"
            use_actor = True

        states_shape = states.shape

        if return_ensemble_result is False:
            assert states_shape[0] == 1, "states.shape[0] must be 1"
            states = states.repeat([self.network_size, 1, 1])
            actions = actions.repeat([self.network_size, 1, 1])

        else:
            if  states_shape[0] != self.network_size:
                states = states.reshape(self.network_size, -1, states_shape[-1])

        result = {}
        for step in range(horizon):
            if use_actor:
                actions, values = actor_critic.act(states, deterministic=False)
            #
            sample_mean, sample_std = vitual_step(states, actions)
            if return_ensemble_result is False:
                _, batch_size, _ = ensemble_model_means.shape
                model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
                batch_idxes = np.arange(0, batch_size)
                samples = ensemble_samples[model_idxes, batch_idxes]
                samples_var = ensemble_model_vars[model_idxes, batch_idxes]

            if return_single is True:

            if self._predict_reward is True:

            if self._predict_cost is True:

            if self._predict_termination is True:


    # pylint: disable-next=too-many-locals
    def mbppo_step(self, obs, act, idx=None, deterministic=False):
        # pylint: disable-next=line-too-long
        """use numpy input to predict single next state by randomly select one model result or select index model result."""
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        if idx is None:
            idx = self.model.elite_model_idxes
        else:
            idx = [idx]
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, self.state_start_dim :] += obs

        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            )

        _, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(idx, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        if self.algo == 'MBPPOLag' and self.model.env_type == 'mujoco-velocity':
            rewards, cost, next_obs = (
                samples[:, 0],
                samples[:, 1],
                samples[:, self.state_start_dim :],
            )
            terminals = self._termination_fn(self.env_name, obs, act, next_obs)
        elif self.algo == 'MBPPOLag' and self.model.env_type == 'gym':
            next_obs = samples
            rewards = None
            cost = None
            terminals = None

        if return_single:
            next_obs = next_obs[0]
            if self.model.env_type == 'mujoco-velocity':
                rewards = rewards[0]
                cost = cost[0]

        return next_obs, rewards, cost, terminals

    # pylint: disable-next=too-many-arguments,too-many-locals
    def safeloop_step(self, obs, act, deterministic=False, all_model=False, repeat_network=False):
        """Use tensor input to predict single next state by randomly select elite model result for online planning"""
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]

        inputs = torch.cat((obs, act), dim=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict_t(
            inputs, repeat_network=repeat_network
        )

        ensemble_model_means[:, :, self.state_start_dim :] += obs

        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + torch.randn(size=ensemble_model_means.shape).to(self.device) * ensemble_model_stds
            )

        # use all dynamics model result
        if all_model:
            samples = ensemble_samples
            samples_var = ensemble_model_vars
        # only use elite model result
        else:
            _, batch_size, _ = ensemble_model_means.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)
            samples = ensemble_samples[model_idxes, batch_idxes]
            samples_var = ensemble_model_vars[model_idxes, batch_idxes]

        return samples, samples_var

    # pylint: disable-next=too-many-arguments, too-many-locals
    def cap_step(self, obs, act, deterministic=False, all_model=True, repeat_network=False):
        """Use tensor input to predict single next state by randomly select elite model result for online planning"""
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]

        inputs = torch.cat((obs, act), dim=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict_t(
            inputs, repeat_network=repeat_network
        )

        ensemble_model_means[:, :, self.state_start_dim :] += obs

        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + torch.randn(size=ensemble_model_means.shape).to(self.device) * ensemble_model_stds
            )

        # use all dynamics model result
        if all_model:
            samples = ensemble_samples
            samples_var = ensemble_model_vars
        # only use elite model result
        else:
            _, batch_size, _ = ensemble_model_means.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)
            samples = ensemble_samples[model_idxes, batch_idxes]
            samples_var = ensemble_model_vars[model_idxes, batch_idxes]

        rewards, rewards_var = samples[:, :, 0].unsqueeze(2), samples_var[:, :, 0].unsqueeze(2)
        next_obs, next_obs_var = (
            samples[:, :, self.state_start_dim :],
            samples_var[:, :, self.state_start_dim :],
        )
        output = {
            'state': (next_obs, next_obs_var),
            'reward': (rewards, rewards_var),
        }
        if self.model.env_type == 'mujoco-velocity':
            cost, cost_var = samples[:, :, 1].unsqueeze(2), samples_var[:, :, 1].unsqueeze(2)
            output['cost'] = (cost, cost_var)

        return output
