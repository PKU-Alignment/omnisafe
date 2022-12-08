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

import itertools
import time
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based import arc
from omnisafe.algorithms.model_based.aux import generate_lidar

### safeloop
from omnisafe.algorithms.model_based.models import core_ac, core_sac
from omnisafe.algorithms.model_based.models.dynamics_predict_env import PredictEnv
from omnisafe.algorithms.model_based.models.dynamicsmodel import EnsembleDynamicsModel

### mbppo and safeloop
from omnisafe.algorithms.model_based.replay_memory import PPOBuffer, ReplayBuffer, SAC_ReplayBuffer
from omnisafe.common.logger import Logger
from omnisafe.utils import distributed_utils
from omnisafe.utils.distributed_utils import proc_id
from omnisafe.utils.tools import get_flat_params_from


def default_termination_function(state, action, next_state):
    '# Default termination function that outputs done=False'
    if torch.is_tensor(next_state):
        done = torch.zeros((next_state.shape[0], 1))
    else:
        done = np.zeros((next_state.shape[0], 1))
    return done


@registry.register
class PolicyGradientModelBased:
    """policy update base class"""

    def __init__(self, env, exp_name, data_dir, seed=0, algo='pg', cfgs=None) -> None:
        self.env = env
        self.env_id = env.env_id
        self.cfgs = deepcopy(cfgs)
        self.exp_name = exp_name
        self.data_dir = data_dir
        self.algo = algo
        self.device = torch.device(self.cfgs['device'])

        # Set up logger and save configuration to disk
        # Get local parameters before logger instance to avoid unnecessary print
        self.params = locals()
        self.params.pop('self')
        self.params.pop('env')
        self.logger = Logger(exp_name=self.exp_name, data_dir=self.data_dir, seed=seed)
        self.logger.save_config(self.params)

        # Set seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.env.reset(seed=seed)
        num_networks = self.cfgs['dynamics_cfgs']['num_networks']
        pred_hidden_size = self.cfgs['dynamics_cfgs']['pred_hidden_size']
        replay_size = self.cfgs['replay_size']

        env_name = 'safepg2'
        self.num_repeat = self.cfgs['action_repeat']

        if self.algo in ['mbppo-lag', 'mbppo_v2']:
            obs_dim = (26,)
            num_elites = self.cfgs['dynamics_cfgs']['num_elites']

            act_dim = self.env.action_space.shape
            self.env_pool = ReplayBuffer(replay_size)
            use_decay = self.cfgs['dynamics_cfgs']['use_decay']
            reward_size = self.cfgs['dynamics_cfgs']['reward_size']
            cost_size = self.cfgs['dynamics_cfgs']['cost_size']
            state_dim = self.cfgs['dynamics_cfgs']['state_dim']
            action_dim = act_dim[0]
            self.cost_criteria = self.cfgs.get('cost_criteria', False)

            self.dynamics = EnsembleDynamicsModel(
                algo,
                num_networks,
                num_elites,
                state_dim,
                action_dim,
                reward_size,
                cost_size,
                pred_hidden_size,
                use_decay=use_decay,
            )
            self.predict_env = PredictEnv(algo, self.dynamics, env_name, 'pytorch')
            # Create actor-critic module
            self.actor_critic = core_ac.MLPActorCritic(
                obs_dim, self.env.action_space, **dict(hidden_sizes=self.cfgs['ac_hidden_sizes'])
            ).to(self.device)
            self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.cfgs['pi_lr'])
            self.vf_optimizer = Adam(self.actor_critic.v.parameters(), lr=self.cfgs['vf_lr'])
            self.cvf_optimizer = Adam(self.actor_critic.vc.parameters(), lr=self.cfgs['vf_lr'])

            self.penalty_param = torch.tensor(
                self.cfgs['lagrange_cfgs']['lagrangian_multiplier_init'],
                device=self.device,
                requires_grad=True,
            ).float()
            self.penalty_optimizer = Adam(
                [self.penalty_param], lr=self.cfgs['lagrange_cfgs']['lambda_lr']
            )

            self.buf = PPOBuffer(
                obs_dim,
                act_dim,
                self.cfgs['On_buffer_cfgs']['local_step_per_epoch'],
                self.cfgs['On_buffer_cfgs']['gamma'],
                self.cfgs['On_buffer_cfgs']['lam'],
            )

            # Set up model saving
            what_to_save = {
                'pi': self.actor_critic.pi,
            }
            self.logger.setup_torch_saver(what_to_save=what_to_save)
            self.logger.torch_save()

        elif self.algo == 'safeloop':
            obs_dim = (26,)
            act_dim = self.env.action_space.shape
            use_decay = self.cfgs['dynamics_cfgs']['use_decay']
            reward_size = self.cfgs['dynamics_cfgs']['reward_size']
            cost_size = self.cfgs['dynamics_cfgs']['cost_size']
            state_dim = self.cfgs['dynamics_cfgs']['state_dim']
            action_dim = act_dim[0]
            num_elites = self.cfgs['dynamics_cfgs']['num_elites']

            self.dynamics = EnsembleDynamicsModel(
                algo,
                num_networks,
                num_elites,
                state_dim,
                action_dim,
                reward_size,
                cost_size,
                pred_hidden_size,
                use_decay=use_decay,
            )
            self.predict_env = PredictEnv(algo, self.dynamics, env_name, 'pytorch')
            ## soft actor critic
            self.replay_buffer = SAC_ReplayBuffer(obs_dim, act_dim, replay_size)
            self.actor_critic = core_sac.MLPActorCritic(
                obs_dim, self.env.action_space, **dict(hidden_sizes=self.cfgs['ac_hidden_sizes'])
            ).to(self.device)
            self.actor_critic_targ = deepcopy(self.actor_critic)
            self.termination_func = default_termination_function
            self.models = self.predict_env
            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.actor_critic_targ.parameters():
                p.requires_grad = False
            # List of parameters for both Q-networks (save this for convenience)
            self.q_params = itertools.chain(
                self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
            )
            # Count variables (protip: try to get a feel for how different size networks behave!)
            self.var_counts = tuple(
                core_sac.count_vars(module)
                for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2]
            )
            # Set up optimizers for policy and q-function
            learning_rate = self.cfgs['sac_lr']
            self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=learning_rate)
            self.q_optimizer = Adam(self.q_params, lr=learning_rate)
            self.v_optimizer = Adam(self.actor_critic.v.parameters(), lr=learning_rate)

            self.automatic_alpha_tuning = True
            if self.automatic_alpha_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.env.action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
                self.alpha = self.log_alpha.exp()
            else:
                self.alpha = self.cfgs['alpha_init']

            self.safeloop_policy = arc.SafeARC(
                state_dim,
                action_dim,
                self.env,
                self.predict_env,
                self.actor_critic,
                default_termination_function,
                device=self.device,
            )

            self.noise_amount = self.cfgs['mpc_config']['exploration_noise']
            self.gamma = self.cfgs['gamma']
            self.polyak = self.cfgs['polyak']
            self.use_bc_loss = self.cfgs['use_bc_loss']

            self.dynamics_freq = self.cfgs['dynamics_freq']
            self.start_timesteps = self.cfgs['start_timesteps']
            self.policy_update_freq = self.cfgs['policy_update_freq']
            self.policy_update_time = self.cfgs['policy_update_time']
            self.batch_size = self.cfgs['batch_size']
            self.eval_freq = self.cfgs['eval_freq']

            self.safeloop_policy.horizon = self.cfgs['mpc_config']['horizon']
            self.safeloop_policy.sol_dim = (
                self.env.action_space.shape[0] * self.cfgs['mpc_config']['horizon']
            )
            self.safeloop_policy.ub = np.repeat(
                self.env.action_space.high, self.cfgs['mpc_config']['horizon'], axis=0
            )
            self.safeloop_policy.lb = np.repeat(
                self.env.action_space.low, self.cfgs['mpc_config']['horizon'], axis=0
            )
            self.safeloop_policy.mean = np.zeros((self.safeloop_policy.sol_dim,))
            self.safeloop_policy.N = self.cfgs['mpc_config']['ARC']['popsize']
            self.safeloop_policy.mixture_coefficient = self.cfgs['mpc_config']['ARC'][
                'mixture_coefficient'
            ]
            self.safeloop_policy.particles = self.cfgs['mpc_config']['ARC']['particles']
            self.safeloop_policy.max_iters = self.cfgs['mpc_config']['ARC']['max_iters']
            self.safeloop_policy.alpha = self.cfgs['mpc_config']['ARC']['alpha']
            self.safeloop_policy.kappa = self.cfgs['mpc_config']['ARC']['kappa']
            if 'reward_horizon' in self.cfgs['mpc_config']['ARC'].keys():
                self.safeloop_policy.reward_horizon = self.cfgs['mpc_config']['ARC'][
                    'reward_horizon'
                ]
            # Set up model saving
            what_to_save = {
                'pi': self.actor_critic.pi,
            }
            self.logger.setup_torch_saver(what_to_save=what_to_save)
            self.logger.torch_save()

        # Setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()

        self.logger.log('Start with training.')

    def _init_mpi(self):
        """
        Initialize MPI specifics
        """
        if distributed_utils.num_procs() > 1:
            # Avoid slowdowns from PyTorch + MPI combo
            distributed_utils.setup_torch_for_mpi()
            datetime = time.time()
            self.logger.log('INFO: Sync actor critic parameters')
            # Sync params across cores: only once necessary, grads are averaged!
            distributed_utils.sync_params(self.actor_critic)
            self.logger.log(f'Done! (took {time.time()-datetime:0.3f} sec.)')

    def algorithm_specific_logs(self, timestep):
        """
        Use this method to collect log information.
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for CPO, etc
        """

    def check_distributed_parameters(self):
        """
        Check if parameters are synchronized across all processes.
        """

        if distributed_utils.num_procs() > 1:
            self.logger.log('Check if distributed parameters are synchronous..')
            modules = {'Policy': self.actor_critic.pi.net, 'Value': self.actor_critic.v.net}
            for key, module in modules.items():
                flat_params = get_flat_params_from(module).numpy()
                global_min = distributed_utils.mpi_min(np.sum(flat_params))
                global_max = distributed_utils.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_v(self, obs, ret):
        """
        computing value loss

        Returns:
            torch.Tensor
        """
        return ((self.actor_critic.v(obs) - ret) ** 2).mean()

    def compute_loss_c(self, obs, ret):
        """
        computing cost loss

        Returns:
            torch.Tensor
        """
        return ((self.actor_critic.c(obs) - ret) ** 2).mean()

    def update_dynamics_model(self):
        """
        training the dynamics model

        Returns:
            No return
        """

    def update_actor_critic(self, data=None):
        """
        update the actor critic

        Returns:
            No return
        """

    def learn(self):
        """learn the policy"""
        if self.algo == 'safeloop':
            self.learn_safeloop()
        elif self.algo == 'mbppo-lag':
            self.learn_mbppo()

    def learn_safeloop(self):
        """training the policy using safeloop"""
        self.start_time = time.time()
        ep_len, ep_ret, ep_cost = 0, 0, 0
        state, done = self.env.reset(), False
        max_real_time_step = int(self.cfgs['max_real_time_step'])
        start_timesteps = int(self.cfgs['start_timesteps'])
        for t in range(0, max_real_time_step, self.num_repeat):  # args.max_timesteps
            ep_len += 1
            if t < start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.safeloop_policy.get_action(np.array(state), env=self.env)
                action = action + np.random.normal(action.shape) * self.noise_amount
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            # Take the safe action
            next_state, reward, done, info = self.env.step(action, self.num_repeat)
            ep_ret += reward
            if 'cost' in info:
                ep_cost += info['cost']

            self.replay_buffer.store(state, action, reward, next_state, done, cost=info['cost'])
            state = next_state

            if t >= start_timesteps and t % self.policy_update_freq == 0:
                for j in range(self.policy_update_time):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update_actor_critic(data=batch)

            if (t + self.num_repeat) % self.dynamics_freq == 0:
                self.update_dynamics_model()

            if done:
                self.safeloop_policy.reset()
                self.logger.store(
                    **{
                        'Metrics/EpRet': ep_ret,
                        'Metrics/EpLen': ep_len,
                        'Metrics/EpCost': ep_cost,
                    }
                )
                ep_ret, ep_cost = 0, 0
                state, done = self.env.reset(), False
                ep_len = 0

            # Evaluate episode
            if (t + self.num_repeat) % self.eval_freq == 0:
                self.log(t)

    def learn_mbppo(self):
        """training the policy using MBPPO-Lag in MBPPO setting safety-gym env"""

        self.start_time = time.time()
        # Main loop: collect experience in env to train dynamics models
        exp_name = 'ppo_test'
        ep_ret, ep_costs, ep_len, violations = 0.0, 0.0, 0, 0
        state, static = self.env.reset()
        hazards_pos = static['hazards']
        mix_real = int(self.cfgs['mixed_real_tiem_step'])
        max_ep_len2 = int(self.cfgs['imaging_time_step'])
        max_training_steps = int(self.cfgs['max_real_time_step'])
        for timestep in range(max_training_steps):
            # generate hazard lidar
            state_vec = generate_lidar(state, hazards_pos)
            state_vec = np.array(state_vec)

            state_tensor = torch.as_tensor(state_vec, device=self.device, dtype=torch.float32)
            action, val, cval, logp = self.actor_critic.step(state_tensor)
            del state_tensor
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            # print(self.env.observation_space.low)
            next_state, reward, done, info = self.env.step(action, self.num_repeat)
            # print("true",next_state,action)
            cost = info['cost']
            if not done and not info['goal_met']:
                self.env_pool.push(
                    state=state,
                    action=action,
                    reward=reward,
                    cost=cost,
                    next_state=next_state,
                    done=done,
                )

            violations += cost
            ep_ret += reward
            ep_costs += cost
            ep_len += 1
            # Mixing some real environment samples
            if timestep % 10000 <= mix_real - 1:
                self.buf.store(
                    obs=state_vec, act=action, rew=reward, crew=cost, val=val, cval=cval, logp=logp
                )

            # logging policy performance in real world
            self.logger.store(**{'Values/V': val, 'Values/C': cval})
            self.logger.store(**{'Values/V': val})
            # Update obs (critical!)
            state = next_state

            timeout = ep_len == self.env.num_steps
            terminal = done or timeout
            epoch_ended = timestep == max_training_steps - 1

            timeout_mixer = ep_len == max_ep_len2
            epoch_ended_mixer = timestep % 10000 == mix_real - 1

            if timestep % 10000 <= mix_real - 1:
                if timeout_mixer or epoch_ended_mixer:
                    state_tensor = torch.as_tensor(
                        state_vec, device=self.device, dtype=torch.float32
                    )
                    _, val, cval, _ = self.actor_critic.step(state_tensor)
                    del state_tensor
                    self.buf.finish_path(val, cval)
                elif done:
                    val = 0
                    cval = 0
                    self.buf.finish_path(val, cval)

            if terminal:
                self.logger.store(
                    **{
                        'Metrics/EpRet': ep_ret,
                        'Metrics/EpLen': ep_len,
                        'Metrics/EpCost': ep_costs,
                    }
                )

            if terminal or epoch_ended:
                state, static = self.env.reset()
                hazards_pos = static['hazards']
                ep_ret, ep_len = 0, 0
                ep_costs = 0

            if (timestep + 1) % 10000 == 0:
                self.update_dynamics_model()
                torch.save(self.dynamics, exp_name + 'env_model.pkl')

                self.update_actor_critic()
                self.log(timestep)

    def log(self, timestep: int):
        """
        logging data
        """
        self.logger.log_tabular('TotalEnvSteps', timestep + self.num_repeat)
        self.logger.log_tabular('Metrics/EpRet')
        self.logger.log_tabular('Metrics/EpCost')
        self.logger.log_tabular('Metrics/EpLen')
        # Some child classes may add information to logs
        self.algorithm_specific_logs(timestep)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.dump_tabular()

    def pre_process_data(self, raw_data: dict):
        """
        Pre-process data, e.g. standardize observations, rescale rewards if
            enabled by arguments.

        Parameters
        ----------
        raw_data
            dictionary holding information obtain from environment interactions

        Returns
        -------
        dict
            holding pre-processed data, i.e. observations and rewards
        """
        data = deepcopy(raw_data)
        # Note: use_reward_scaling is currently applied in Buffer...
        # If self.use_reward_scaling:
        #     rew = self.actor_critic.ret_oms(data['rew'], subtract_mean=False, clip=True)
        #     data['rew'] = rew

        if self.cfgs['standardized_obs']:
            assert 'obs' in data
            obs = data['obs']
            data['obs'] = self.actor_critic.obs_oms(obs, clip=False)
        return data

    def update_running_statistics(self, data):
        """
        Update running statistics, e.g. observation standardization,
        or reward scaling. If MPI is activated: sync across all processes.
        """
        if self.cfgs['standardized_obs']:
            self.actor_critic.obs_oms.update(data['obs'])

        # Apply Implement Reward scaling
        if self.cfgs['scale_rewards']:
            self.actor_critic.ret_oms.update(data['discounted_ret'])
