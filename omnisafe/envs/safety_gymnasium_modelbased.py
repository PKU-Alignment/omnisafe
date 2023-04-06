# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Environments in the Safety Gymnasium."""


from typing import Any, Dict, Optional, Tuple, Union


import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register

import gymnasium

@env_register
class SafetyGymnasiumModelBased(CMDP):
    """Safety Gymnasium environment."""

    _support_envs = [
        'SafetyPointGoal0-v0-modelbased',
        'SafetyPointGoal1-v0-modelbased',
        'SafetyPointGoal2-v0-modelbased',
        'SafetyCarGoal0-v0-modelbased',
        'SafetyCarGoal1-v0-modelbased',
        'SafetyCarGoal2-v0-modelbased',
        'SafetyAntGoal0-v0-modelbased',
        'SafetyAntGoal1-v0-modelbased',
        'SafetyAntGoal2-v0-modelbased',

    ]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, num_envs: int = 1, **kwargs) -> None:
        super().__init__(env_id)
        if num_envs > 1:
            self._env = safety_gymnasium.vector.make(env_id=env_id.replace('-modelbased', ''), num_envs=num_envs, **kwargs)
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            self._env = safety_gymnasium.make(id=env_id.replace('-modelbased', ''), autoreset=False, **kwargs)
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space

        self._num_envs = num_envs
        self._metadata = self._env.metadata
        if env_id in [
            'SafetyPointGoal1-v0-modelbased',
            'SafetyPointGoal2-v0-modelbased',
            'SafetyCarGoal1-v0-modelbased',
            'SafetyCarGoal2-v0-modelbased',
            'SafetyAntGoal1-v0-modelbased',
            'SafetyAntGoal2-v0-modelbased',
        ]:
            self._constraints = ['hazards'] #'gremlins', 'buttons'],
            self._xyz_sensors = ['velocimeter', 'accelerometer']
            self._angle_sensors = ['gyro', 'magnetometer']
            self._flatten_order = (
                self._xyz_sensors + self._angle_sensors + ['goal'] + self._constraints + ['robot_m'] + ['robot']
            )
            self._base_state = self._xyz_sensors + self._angle_sensors
            self._task = 'Goal'
            self._env.reset()
            self.goal_position = self._env.task.goal.pos
            self.robot_position = self._env.task.agent.pos
            self.hazards_position = self._env.task.hazards.pos
            self.goal_distance = self._dist_xy(self.robot_position, self.goal_position)

            coordinate_sensor_obs = self._get_coordinate_sensor()
            self._coordinate_obs_size = sum(np.prod(i.shape) for i in list(coordinate_sensor_obs.values()))
            offset = 0
            self.key_to_slice = {}
            for k in self._flatten_order:
                k_size = np.prod(coordinate_sensor_obs[k].shape)
                self.key_to_slice[k] = slice(offset, offset + k_size)

                offset += k_size
            self._base_state_size = sum(np.prod(coordinate_sensor_obs[k].shape) for k in list(self._base_state))
            self.key_to_slice['base_state'] = slice(0, self._base_state_size)
            self._num_lidar_bin = 16
            self._max_lidar_dist = 3
            self.original_observation_space = self.observation_space
            self.coordinate_observation_space = gymnasium.spaces.Box(
                            -np.inf, np.inf, (self._coordinate_obs_size,), dtype=np.float32
                        )# 26
            flat_coordinate_obs = self._get_flat_coordinate(coordinate_sensor_obs)
            self.lidar_observation_space = gymnasium.spaces.Box(
                            -np.inf, np.inf, (self._get_lidar_from_coordinate(flat_coordinate_obs).shape[0], ), dtype=np.float32
                        )# 26



        else:
            self._task = None
            raise NotImplementedError

    def get_cost_from_coordinate(self, state):
        assert state.shape == (self.coordinate_observation_space.shape[0],)
        robot_pos = state[self.key_to_slice['robot']]
        # ----cost----
        cost = 0
        hazards_cost = 1.0
        for h_pos in self.hazards_position:
            h_dist = self._dist_xy(h_pos, robot_pos)
            if h_dist <= self.hazards_size:
                cost += hazards_cost * (self.hazards_size - h_dist)
        if cost > 0:
            cost = 1
        else:
            cost = 0
        cost = torch.as_tensor(
                cost, dtype=torch.float32
            )
        return cost

    def get_reward_from_coordinate(self, state):
        assert state.shape == (self.coordinate_observation_space.shape[0],)
        last_dist_goal = self.goal_distance
        robot_pos = state[self.key_to_slice['robot']]
        reward = 0
        reward_distance = 1.0
        reward_goal = 1.0
        goal_size = 0.3
        dist_goal = self._dist_xy(robot_pos, self.goal_position)
        reward += (last_dist_goal - dist_goal) * reward_distance
        last_dist_goal = dist_goal
        goal_flag = False
        if dist_goal < goal_size:
            reward += reward_goal
            goal_flag = True
        # clip reward
        if reward < -10:
            reward = -10
        elif reward > 10:
            reward = 10
        self.goal_distance = last_dist_goal
        reward = torch.as_tensor(
                reward, dtype=torch.float32
            )
        goal_flag = torch.as_tensor(
                reward, dtype=torch.float32
            )
        return reward, goal_flag

    def get_reward_cost(self, state):
        '''Assuming we have reward & cost function. available with us in closed form.'''
        last_dist_goal = self.goal_distance
        robot_pos = state[self.key_to_slice['robot']]
        # ----cost----
        cost = 0
        hazards_cost = 1.0
        for h_pos in self.hazards_position:
            h_dist = self.dist_xy(h_pos, robot_pos)
            if h_dist <= self.hazards_size:
                cost += hazards_cost * (self.hazards_size - h_dist)
        if cost > 0:
            cost = 1
        else:
            cost = 0
        # ----reward----

        reward = 0
        reward_distance = 1.0
        reward_goal = 1.0
        goal_size = 0.3

        dist_goal = self.dist_xy(robot_pos, self.goal_position)
        reward += (last_dist_goal - dist_goal) * reward_distance
        last_dist_goal = dist_goal
        goal_flag = False
        if dist_goal < goal_size:
            reward += reward_goal
            goal_flag = True
        # clip reward
        if reward < -10:
            reward = -10
        elif reward > 10:
            reward = 10
        self.goal_distance = last_dist_goal
        return reward, cost, goal_flag


    def _get_lidar_from_coordinate(self, obs):
        """Get lidar observation"""
        robot_matrix_x_y = obs[self.key_to_slice['robot_m']]
        robot_matrix_x = robot_matrix_x_y[0]
        robot_matrix_y = robot_matrix_x_y[1]
        first_row = [robot_matrix_x, robot_matrix_y, 0]
        second_row = [-robot_matrix_y, robot_matrix_x, 0]
        third_row = [0, 0, 1]
        robot_matrix = [first_row, second_row, third_row]
        robot_pos = obs[self.key_to_slice['robot']]
        hazards_lidar_vec = self._obs_lidar_pseudo(robot_matrix, robot_pos, self.hazards_position)

        goal_lidar_vec = self._obs_lidar_pseudo(robot_matrix, robot_pos, [self.goal_position])
        base_state_vec = obs[self.key_to_slice['base_state']]

        obs_vec = list(base_state_vec) + list(hazards_lidar_vec) + list(goal_lidar_vec)

        #obs_vec = self.make_observation(obs, lidar_vec)
        obs_vec = np.array(obs_vec)
        return obs_vec

    def _ego_xy(self, robot_matrix, robot_pos, pos):
        '''Return the egocentric XY vector to a position from the robot'''
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = robot_pos
        robot_mat = robot_matrix

        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        robot_3vec = np.concatenate([robot_3vec, [0]])
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]


    def _obs_lidar_pseudo(
        self, robot_matrix, robot_pos, positions
    ):  # pylint: disable=too-many-locals
        '''
        Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        '''
        obs = np.zeros(self._num_lidar_bin)
        lidar_exp_gain = 1.0
        lidar_alias = True
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            position_z = np.complex(
                *self._ego_xy(robot_matrix, robot_pos, pos)
            )  # X, Y as real, imaginary components
            dist = np.abs(position_z)
            angle = np.angle(position_z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self._num_lidar_bin
            sensor_bin = int(angle / bin_size)
            bin_angle = bin_size * sensor_bin
            if self._max_lidar_dist is None:
                sensor = np.exp(-lidar_exp_gain * dist)
            else:
                sensor = max(0, self._max_lidar_dist - dist) / self._max_lidar_dist
            obs[sensor_bin] = max(obs[sensor_bin], sensor)
            # Aliasing
            if lidar_alias:
                alias = (angle - bin_angle) / bin_size
                assert (
                    0 <= alias <= 1
                ), f'bad alias {alias}, dist {dist}, angle {angle}, bin {sensor_bin}'
                bin_plus = (sensor_bin + 1) % self._num_lidar_bin
                bin_minus = (sensor_bin - 1) % self._num_lidar_bin
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def _get_flat_coordinate(self, coordinate_obs) -> np.ndarray:
        '''get the flattened obs.'''
        flat_obs = np.zeros(self.coordinate_observation_space.shape[0])
        for k in self._flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = coordinate_obs[k].flat
        return flat_obs

    def _get_coordinate_sensor(self) -> dict:
        '''
        We will ignore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self._env.task.agent.pos
        goal_pos = self._env.task.goal.pos
        vases_pos_list = self._env.task.vases.pos  # list of shape (3,) ndarray
        hazards_pos_list = self._env.task.hazards.pos  # list of shape (3,) ndarray
        ego_goal_pos = self._env.task._ego_xy(goal_pos[:2])
        ego_vases_pos_list = [
            self._env.task._ego_xy(pos[:2]) for pos in vases_pos_list
        ]  # list of shape (2,) ndarray
        ego_hazards_pos_list = [
            self._env.task._ego_xy(pos[:2]) for pos in hazards_pos_list
        ]  # list of shape (2,) ndarray

        # append obs to the dict
        for sensor in self._xyz_sensors:  # Explicitly listed sensors
            if sensor == 'accelerometer':
                obs[sensor] = self._env.task.agent.get_sensor(sensor)[:1]  # only x axis matters
            elif sensor == 'ballquat_rear':
                obs[sensor] = self._env.task.agent.get_sensor(sensor)
            else:
                obs[sensor] = self._env.task.agent.get_sensor(sensor)[:2]  # only x,y axis matters

        for sensor in self._angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self._env.task.agent.get_sensor(sensor)[
                    2:
                ]  # [2:] # only z axis matters
                # pass # gyro does not help
            else:
                obs[sensor] = self._env.task.agent.get_sensor(sensor)
        # --------modification-----------------
        obs['robot'] = np.array(robot_pos[:2])
        obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
        robot_matrix = self._env.task.agent.mat
        obs['robot_m'] = np.array(robot_matrix[0][:2])
        obs['goal'] = ego_goal_pos  # (2,)
        #obs['vases'] = np.array(ego_vases_pos_list)  # (vase_num, 2)
        return obs


    def _dist_xy(
            self,
            pos1: Union[np.ndarray, list],
            pos2: Union[np.ndarray, list],
            ) -> float:
        '''Return the distance from the robot to an XY position.'''
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        if pos1.shape == (3,):
            pos1 = pos1[:2]
        if pos2.shape == (3,):
            pos2 = pos2[:2]
        return np.sqrt(np.sum(np.square(pos1 - pos2)))

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs_original, reward, cost, terminated, truncated, info = self._env.step(action)
        if self._task == 'Goal':
            self.goal_position = self._env.task.goal.pos
            self.robot_position = self._env.task.agent.pos
            self.hazards_position = self._env.task.hazards.pos
            self.goal_distance = self._dist_xy(self.robot_position, self.goal_position)
            coordinate_sensor_obs = self._get_coordinate_sensor()
            obs = self._get_flat_coordinate(coordinate_sensor_obs)

            info['obs_original'] = obs_original
            goal_met = 'goal_met' in info.keys()  # reach the goal
            info['goal_met'] = goal_met


        obs, reward, cost, terminated, truncated = map(
            lambda x: torch.as_tensor(x, dtype=torch.float32),
            (obs, reward, cost, terminated, truncated),
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ]
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'], dtype=torch.float32
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        obs_original, info = self._env.reset(seed=seed)
        if self._task == 'Goal':
            self.goal_position = self._env.task.goal.pos
            self.robot_position = self._env.task.agent.pos
            self.hazards_position = self._env.task.hazards.pos
            self.goal_distance = self._dist_xy(self.robot_position, self.goal_position)
            coordinate_sensor_obs = self._get_coordinate_sensor()
            obs = self._get_flat_coordinate(coordinate_sensor_obs)
            info['obs_original'] = obs_original
            info['goal_met'] = False
        return torch.as_tensor(obs, dtype=torch.float32), info

    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        return torch.as_tensor(self._env.action_space.sample(), dtype=torch.float32)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
