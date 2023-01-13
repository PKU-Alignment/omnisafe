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
"""Environment wrapper for on-policy algorithms."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import safety_gymnasium
import torch
from safety_gymnasium.bases import BaseTask

from omnisafe.common.base_buffer import BaseBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.common.record_queue import RecordQueue
from omnisafe.common.vector_buffer import VectorBuffer as Buffer
from omnisafe.models import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.typing import Dict, NamedTuple, Optional, Tuple, Union
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import expand_dims
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@dataclass
class RenderData:
    """Data for rendering."""

    env_id: str
    render_mode: str
    camera_id: int
    camera_name: str
    width: int
    height: int


@dataclass
class RolloutLog:
    """Log for roll out."""

    ep_ret: np.ndarray
    ep_costs: np.ndarray
    ep_len: np.ndarray


@dataclass
class RolloutData:
    """Data for roll out."""

    local_steps_per_epoch: int
    max_ep_len: int
    use_cost: bool
    current_obs: np.ndarray
    rollout_log: RolloutLog


@WRAPPER_REGISTRY.register
class CMDPWrapper:  # pylint: disable=too-many-instance-attributes
    """Implementation of the environment wrapper for on-policy algorithms.
    ``omnisafe`` use different environment wrappers for different kinds of algorithms.
    This is the environment wrapper for on-policy algorithms.
    .. list-table:: Environment Wrapper for Different Algorithms
        * - Algorithm
        - Environment Wrapper
        - Function
        * - On-Policy
        - :class:`OnPolicyEnvWrapper`
        - On-policy wrapper :meth:`rollout` for each episode,
            log useful information and collect data for training.
        * - Off-Policy
        - :class:`OffPolicyEnvWrapper`
        - Off-policy wrapper :meth:`rollout` for each episode,
            log useful information and collect data for training.
            Off-policy algorithms need to update the policy network before the end of each episode.
        * - Saute
        - :class:`SauteEnvWrapper`
        - Saute wrapper additionally maintains a ``safety state`` for each episode.
            If the safety state is ``safe``, the ``reward`` is the original reward.
            If the safety state is ``unsafe``, the ``reward`` is the ``unsafe reward`` (always 0).
        * - Simmer
        - :class:`SimmerEnvWrapper`
        - Simmer wrapper also maintains a ``safety state`` for each episode.
            Additionally, it uses a ``PID controller`` and ``Q controller`` to control the ``safety state``.
        * - Early Terminated
        - :class:`EarlyTerminatedEnvWrapper`
        - Early terminated wrapper stop the episode when the ``cost`` is not 0.
        * - Safety Layer
        - :class:`SafetyLayerEnvWrapper`
        - Safety layer wrapper pre-train a ``safety layer`` to control the ``action``.
    """

    def __init__(self, env_id, cfgs: Optional[NamedTuple] = None, **env_kwargs) -> None:
        """Initialize environment wrapper.
        Args:
            env_id (str): environment id.
            cfgs (collections.namedtuple): configs.
            env_kwargs (dict): The additional parameters of environments.
        """
        # self.env = gymnasium.make(env_id, **env_kwargs)
        self.cfgs = deepcopy(cfgs)
        self.env = None
        self.action_space = None
        self.observation_space = None
        self.make(env_id, env_kwargs)
        if distributed_utils.num_procs() == 1:
            torch.set_num_threads(self.cfgs.num_threads)
        width = self.env.width if hasattr(self.env, 'width') else 256
        height = self.env.height if hasattr(self.env, 'height') else 256
        self.render_data = RenderData(
            env_id,
            env_kwargs.get('render_mode', None),
            env_kwargs.get('camera_id', None),
            env_kwargs.get('camera_name', None),
            width,
            height,
        )
        if hasattr(self.env, '_max_episode_steps'):
            max_ep_len = self.env._max_episode_steps
        else:
            max_ep_len = 1000
        self.rollout_data = RolloutData(
            0.0,
            max_ep_len,
            False,
            None,
            RolloutLog(
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
                np.zeros(self.cfgs.num_envs),
            ),
        )
        self.set_seed(int(self.cfgs.env_seed) + 10000 * distributed_utils.proc_id())
        self.obs_normalizer = (
            Normalizer(shape=(self.cfgs.num_envs, self.observation_space.shape[0]), clip=5)
            if self.cfgs.normalized_obs
            else None
        )
        self.rew_normalizer = (
            Normalizer(shape=(self.cfgs.num_envs, 1), clip=5) if self.cfgs.normalized_rew else None
        )
        self.cost_normalizer = (
            Normalizer(shape=(self.cfgs.num_envs, 1), clip=5) if self.cfgs.normalized_cost else None
        )
        self.record_queue = RecordQueue('ep_ret', 'ep_cost', 'ep_len', maxlen=self.cfgs.max_len)
        self.rollout_data.current_obs = CMDPWrapper.reset(self)[0]

    def make(self, env_id, env_kwargs) -> BaseTask:
        """Create environments"""
        if self.cfgs.num_envs == 1:
            self.env = safety_gymnasium.make(env_id, **env_kwargs)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        else:
            self.env = safety_gymnasium.vector.make(
                env_id, asynchronous=self.cfgs.async_env, num_envs=self.cfgs.num_envs, **env_kwargs
            )
            self.observation_space = self.env.single_observation_space
            self.action_space = self.env.single_action_space

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment.
        At the end of each episode, the environment will be reset.
        Args:
            seed (int): random seed.
        """
        obs, info = self.env.reset()
        if self.cfgs.num_envs == 1:
            obs, info = expand_dims(obs, info)
        return obs, info

    def set_seed(self, seed: Union[int, Tuple]) -> None:
        """Set random seed.

        Args:
            seed (int): random seed.
        """
        self.env.reset(seed=seed)

    def sample_action(self):
        """Sample action from the environment."""
        return self.env.action_space.sample()

    def render(self):
        """render the vectored environment."""
        return self.env.render()

    def set_rollout_cfgs(self, **kwargs: dict) -> None:
        """Set rollout configs

        .. note::
            Current On-Policy algorithms does not need to set the configs.
        If you implement a new algorithm and need to set the :class:`OnPolicyEnvWrapper` configs,
        for example, ``deteministic`` and ``local_steps_per_epoch``,
        you can just:

        .. code-block:: python
            :linenos:
            set_rollout_data(deteministic=True, local_steps_per_epoch=1000)

        Args:
            kwargs (dict): rollout configs.
        """
        for key, value in kwargs.items():
            setattr(self.rollout_data, key, value)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Step the environment.

        The environment will be stepped by the action from the agent.
        Corresponding to the Markov Decision Process,
        the environment will return the ``next observation``,
        ``reward``, ``cost``, ``terminated``, ``truncated`` and ``info``.

        Args:
            action (np.ndarray): action.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action.squeeze())
        if self.cfgs.num_envs == 1:
            next_obs, reward, cost, terminated, truncated, info = expand_dims(
                next_obs, reward, cost, terminated, truncated, info
            )
            if terminated | truncated:
                next_obs, info = self.reset()
        self.rollout_data.rollout_log.ep_ret += reward
        self.rollout_data.rollout_log.ep_costs += cost
        self.rollout_data.rollout_log.ep_len += np.ones(self.cfgs.num_envs)
        return next_obs, reward, cost, terminated, truncated, info

    # pylint: disable-next=too-many-locals
    def on_policy_roll_out(
        self,
        agent: Union[ConstraintActorCritic, ConstraintActorQCritic],
        buf: Buffer,
        logger: Logger,
    ) -> None:
        """Collect data and store to experience buffer.

        :meth:`roll_out` is the main function of the environment wrapper.
        It will collect data from the environment and store to the experience buffer.

        .. note::
            In each step,
            - the environment will be stepped by the action from the agent.
            - Then the data will be stored to the experience buffer.
            - Finally the logger will log useful information.

        Args:
            agent (torch.nn.Module): agent.
            buf (Buffer): experience buffer.
            logger (Logger): logger.
        """
        obs, _ = self.reset()
        for step_i in range(self.rollout_data.local_steps_per_epoch):
            if self.cfgs.normalized_obs:
                # Note: do the updates at the end of batch!
                obs = self.obs_normalizer.normalize(obs)
            action, value, cost_value, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, done, truncated, _ = self.step(action)
            if self.cfgs.normalized_rew:
                # Note: do the updates at the end of batch!
                reward = self.rew_normalizer.normalize(reward)
            if self.cfgs.normalized_cost:
                # Note: do the updates at the end of batch!
                cost = self.cost_normalizer.normalize(cost)

            # Save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buffer
            buf.store(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=logp,
                cost=cost,
                cost_val=cost_value,
            )

            # Store values for statistic purpose
            if self.rollout_data.use_cost:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})

            # Update observation
            obs = next_obs

            terminals = done | truncated
            epoch_ended = step_i == self.rollout_data.local_steps_per_epoch - 1
            for idx, terminal in enumerate(terminals):
                timeout = self.rollout_data.rollout_log.ep_len[idx] == self.rollout_data.max_ep_len
                terminal = terminal or timeout
                if terminal or epoch_ended:
                    if timeout or epoch_ended:
                        _, terminal_value, terminal_cost_value, _ = agent.step(
                            torch.as_tensor(obs[idx], dtype=torch.float32)
                        )
                    else:
                        terminal_value, terminal_cost_value = 0.0, 0.0
                    buf.finish_path(
                        last_val=terminal_value,
                        last_cost_val=terminal_cost_value,
                        idx=idx,
                    )
                if terminal and not epoch_ended:
                    # Log info about epoch
                    self.rollout_log(logger, idx)
                # Only save EpRet / EpLen if trajectory finished

    # pylint: disable-next=too-many-arguments, too-many-locals
    def off_policy_roll_out(
        self,
        agent: Union[ConstraintActorCritic, ConstraintActorQCritic],
        buf: BaseBuffer,
        logger: Logger,
        deterministic: bool,
        use_rand_action: bool,
        ep_steps: int,
    ) -> None:
        """Collect data and store to experience buffer.

        :meth:`roll_out` is the main function of the environment wrapper.
        It will collect data from the environment and store to the experience buffer.

        .. note::
            In each step,
            - the environment will be stepped by the action from the agent.
            - Then the data will be stored to the experience buffer.
            - The logger will store the useful information.
            - Remember the current state and action.

            Recall them after updating the policy network.

        Args:
            agent (torch.nn.Module): agent.
            buf (Buffer): experience buffer.
            logger (Logger): logger.
        """
        for _ in range(ep_steps):
            obs = self.rollout_data.current_obs
            if self.cfgs.normalized_obs:
                # Note: do the updates at the end of batch!
                obs = self.obs_normalizer.normalize(obs)
            action, value, cost_value, _ = agent.step(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic
            )
            # Store values for statistic purpose
            if self.rollout_data.use_cost:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})
            if use_rand_action:
                action = self.env.action_space.sample()
            # Step the env
            next_obs, reward, cost, done, truncated, _ = self.step(action)
            if self.cfgs.normalized_rew:
                # Note: do the updates at the end of batch!
                reward = self.rew_normalizer.normalize(reward)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            self.rollout_data.current_obs = next_obs
            if self.cfgs.normalized_obs:
                # Note: do the updates at the end of batch!
                next_obs = self.obs_normalizer.normalize(next_obs)
            terminals = done | truncated
            epoch_ended = self.rollout_data.rollout_log.ep_len >= self.rollout_data.max_ep_len
            terminals = terminals & ~epoch_ended
            buf.store(obs, action, reward, cost, next_obs, terminals)
            for idx, terminal in enumerate(terminals):
                if terminal or epoch_ended[idx]:
                    self.rollout_log(logger=logger, idx=idx)

    def rollout_log(
        self,
        logger,
        idx,
    ) -> None:
        """Log the information of the rollout."""
        self.record_queue.append(
            ep_ret=self.rollout_data.rollout_log.ep_ret[idx],
            ep_cost=self.rollout_data.rollout_log.ep_costs[idx],
            ep_len=self.rollout_data.rollout_log.ep_len[idx],
        )
        avg_ep_ret, avg_ep_cost, avg_ep_len = self.record_queue.get_mean(
            'ep_ret', 'ep_cost', 'ep_len'
        )
        logger.store(
            **{
                'Metrics/EpRet': avg_ep_ret,
                'Metrics/EpCost': avg_ep_cost,
                'Metrics/EpLen': avg_ep_len,
            }
        )
        (
            self.rollout_data.rollout_log.ep_ret[idx],
            self.rollout_data.rollout_log.ep_costs[idx],
            self.rollout_data.rollout_log.ep_len[idx],
        ) = (0.0, 0.0, 0.0)
