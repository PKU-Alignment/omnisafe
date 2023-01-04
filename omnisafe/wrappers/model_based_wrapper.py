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
"""Environment wrapper for model-based algorithms."""

import gymnasium
import numpy as np
import safety_gymnasium
import torch

from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


# ----------------------------------------------------------------------------------------------------------
ROBOTS = ['Point', 'Car', 'Doggo']
TASKS = ['Goal', 'Button']

XYZ_SENSORS = dict(
    Point=['velocimeter'],
    Car=['velocimeter'],
    Doggo=['velocimeter', 'accelerometer'],
)

ANGLE_SENSORS = dict(
    Point=['gyro', 'magnetometer'], Car=['magnetometer', 'gyro'], Doggo=['magnetometer', 'gyro']
)

CONSTRAINTS_SAFELOOP = dict(
    Goal=['vases', 'hazards'],
    Button=['hazards', 'gremlins', 'buttons'],
)
CONSTRAINTS_MBPPO = dict(
    Goal=['hazards'],
    Button=['hazards', 'gremlins', 'buttons'],
)


@WRAPPER_REGISTRY.register
class ModelBasedEnvWrapper:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Model-based Environment"""

    def __init__(self, algo, env_id, render_mode=None):
        self.algo = algo
        self.env_id = env_id  # safety gym not use this attribute
        self.render_mode = render_mode
        self.timestep = 0
        self.num_steps = 1000
        self.goal_distance = 0
        self.modelbased_safetygym = [
            'SafetyPointGoal3-v0',
            'SafetyCarGoal3-v0',
            'SafetyPointGoal1-v0',
            'SafetyCarGoal1-v0',
        ]
        self.modelbased_mujoco_velocity = [
            'Ant-v4',
            'Swimmer-v4',
            'HalfCheetah-v4',
            'Hopper-v4',
            'Humanoid-v4',
            'Walker2d-v4',
            'Ant-v3',
            'Swimmer-v3',
            'HalfCheetah-v3',
            'Hopper-v3',
            'Humanoid-v3',
            'Walker2d-v3',
        ]
        assert (
            env_id in self.modelbased_safetygym + self.modelbased_mujoco_velocity
        ), f'not support {env_id}'
        if env_id in self.modelbased_safetygym:
            self.robot = 'Point' if 'Point' in env_id else 'Car'
            self.task = 'Goal'
            self.env_type = 'gym'
            self.hazards_size = 0.2
            self.robot = self.robot.capitalize()  # mujoco  not use this attribute
            self.task = self.task.capitalize()  # mujoco  not use this attribute
            assert self.robot in ROBOTS, f'can not recognize the robot type {self.robot}'
            assert self.task in TASKS, f'can not recognize the task type {self.task}'
            self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
            self.init_sensor()
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.ac_state_size,), dtype=np.float32
            )
            self.action_space = gymnasium.spaces.Box(
                -1, 1, (self.env.action_space.shape[0],), dtype=np.float32
            )
            self.goal_position = self.env.task.goal_pos[0][:2]
            self.robot_position = self.env.task.robot_pos
            self.hazards_position = self.env.task.hazards_pos
        elif env_id in self.modelbased_mujoco_velocity:
            self.env_type = 'mujoco-velocity'
            self.env = gymnasium.make(env_id)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.dynamics_state_size = self.observation_space.shape[0]
            self.ac_state_size = self.observation_space.shape[0]

    def set_eplen(self, eplen):
        """Set episode length"""
        self.num_steps = eplen

    def get_cost_from_obs(self, obs, is_binary):
        """Get batch cost from batch observation"""
        if torch.is_tensor(obs):
            obs = obs.cpu().detach().numpy()
        batch_size = obs.shape[0]
        hazards_key = self.key_to_slice['hazards']
        hazard_obs = obs[:, hazards_key].reshape(batch_size, -1, 2)
        hazards_dist = np.sqrt(np.sum(np.square(hazard_obs), axis=2)).reshape(batch_size, -1)
        if is_binary:
            cost = np.where(hazards_dist <= self.hazards_size, 1.0, 0.0)
            cost = cost.sum(1)
            cost = np.where(cost >= 1, 1.0, 0.0)
        else:
            cost = ((hazards_dist < self.hazards_size) * (self.hazards_size - hazards_dist)).sum(
                1
            ) * 10
        return cost

    def init_sensor(self):
        """Initialize sensor observation"""
        self.xyz_sensors = XYZ_SENSORS[self.robot]
        self.angle_sensors = ANGLE_SENSORS[self.robot]
        self.constraints_safeloop = CONSTRAINTS_SAFELOOP[self.task]
        self.constraints_mbppo = CONSTRAINTS_MBPPO[self.task]
        self.base_state_name = self.xyz_sensors + self.angle_sensors + ['goal']
        self.env.reset()
        obs = self.get_obs()
        self.obs_flat_size = sum(np.prod(i.shape) for i in list(obs.values()))
        if self.algo == 'MBPPOLag':
            self.flatten_order = (
                self.base_state_name + self.constraints_mbppo + ['robot_m'] + ['robot']
            )
        elif self.algo in ['SafeLOOP', 'CAP']:
            self.flatten_order = self.base_state_name + self.constraints_safeloop

        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)

            offset += k_size
        self.base_state_dim = sum(np.prod(obs[k].shape) for k in list(self.base_state_name))
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice['base_state'] = slice(0, self.base_state_dim)

        self.reset()
        obs_flat = self.get_obs_flatten()
        if self.algo == 'MBPPOLag':
            self.dynamics_state_size = obs_flat.shape[0]  # 42
            self.ac_state_size = np.array(self.generate_lidar(obs_flat)).shape[0]  # 26

        elif self.algo in ['SafeLOOP', 'CAP']:
            self.dynamics_state_size = obs_flat.shape[0]  # 42
            self.ac_state_size = obs_flat.shape[0]  # 42

    def reset(self):
        """Reset Environment"""
        self.timestep = 0  # Reset internal timer

        if self.env_type == 'mujoco-velocity':
            obs, _ = self.env.reset()
            return obs

        self.env.reset()
        obs = self.get_obs_flatten()
        if self.algo == 'MBPPOLag':
            self.goal_position = self.env.task.goal_pos[0][:2]
            self.robot_position = self.env.task.robot_pos
            self.hazards_position = self.env.task.hazards_pos
            self.goal_distance = self.dist_xy(self.robot_position, self.goal_position)

        return obs

    def step(self, action, num_repeat):  # pylint: disable=too-many-locals
        """Simulate Environment"""
        reward = 0
        cost = 0
        step_num = 0
        if self.env_type == 'gym':
            for _ in range(num_repeat):
                control = action
                _, reward_k, cost_k, terminated, truncated, info = self.env.step(control)
                terminated = False  # not used now
                step_num += 1
                reward += reward_k
                cost += cost_k
                self.timestep += 1  # Increment internal timer
                if self.timestep >= self.num_steps:
                    truncated = True
                observation = self.get_obs_flatten()
                goal_met = 'goal_met' in info.keys()  # reach the goal
                if terminated or truncated or goal_met:
                    # the action is not related to next state, so break
                    break
            if self.algo in ['MBPPOLag', 'SafeLOOP', 'CAP']:
                info = {'cost': cost, 'goal_met': goal_met, 'step_num': step_num}
        elif self.env_type == 'mujoco-velocity':
            for _ in range(num_repeat):
                control = action
                state_k, reward_k, terminated, truncated, info = self.env.step(control)
                step_num += 1
                reward += reward_k
                if 'y_velocity' not in info:
                    cost_k = np.abs(info['x_velocity'])
                else:
                    cost_k = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                cost += cost_k
                self.timestep += 1  # Increment internal timer
                if self.timestep >= self.num_steps:
                    truncated = True
                if terminated or truncated:
                    # the action is not related to next state, so break
                    break
            info = {'cost': cost, 'goal_met': False, 'step_num': step_num}
            observation = state_k
        return observation, reward, cost, terminated, truncated, info

    def render(self):
        """render environment"""
        return self.env.render()

    def close(self):
        """close environment"""
        self.env.close()

    def recenter(self, pos):
        '''Return the egocentric XY vector to a position from the robot'''
        return self.env.task.ego_xy(pos)

    def get_obs(self):
        '''
        We will ignore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.task.robot_pos
        goal_pos = self.env.task.goal_pos[0]
        vases_pos_list = self.env.task.vases_pos  # list of shape (3,) ndarray
        hazards_pos_list = self.env.task.hazards_pos  # list of shape (3,) ndarray
        ego_goal_pos = self.recenter(np.array(goal_pos[:2]))
        ego_vases_pos_list = [
            self.env.task.ego_xy(pos[:2]) for pos in vases_pos_list
        ]  # list of shape (2,) ndarray
        ego_hazards_pos_list = [
            self.env.task.ego_xy(pos[:2]) for pos in hazards_pos_list
        ]  # list of shape (2,) ndarray

        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor == 'accelerometer':
                obs[sensor] = self.env.task.world.get_sensor(sensor)[:1]  # only x axis matters
            elif sensor == 'ballquat_rear':
                obs[sensor] = self.env.task.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.task.world.get_sensor(sensor)[:2]  # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.task.world.get_sensor(sensor)[
                    2:
                ]  # [2:] # only z axis matters
                # pass # gyro does not help
            else:
                obs[sensor] = self.env.task.world.get_sensor(sensor)
        if self.algo == 'MBPPOLag':
            # --------modification-----------------
            obs['robot'] = np.array(robot_pos[:2])
            obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
            robot_matrix = self.env.task.world.robot_mat()
            obs['robot_m'] = np.array(robot_matrix[0][:2])
            obs['goal'] = ego_goal_pos  # (2,)
        elif self.algo in ['CAP', 'SafeLOOP']:
            obs['vases'] = np.array(ego_vases_pos_list)  # (vase_num, 2)
            obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
            obs['goal'] = ego_goal_pos  # (2,)
        return obs

    def get_obs_flatten(self):
        '''get the flattened obs.'''
        obs = self.get_obs()
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = obs[k].flat
        return flat_obs

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.task.dist_goal()

    @property
    def action_range(self):
        """Get action range"""
        return float(self.env.action_space.low[0]), float(self.env.action_space.high[0])

    def sample_random_action(self):
        '''Sample an action randomly from a uniform distribution over all valid actions.'''
        return self.env.action_space.sample()

    def dist_xy(self, pos1, pos2):
        '''Return the distance from the robot to an XY position.'''
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        if pos1.shape == (3,):
            pos1 = pos1[:2]
        if pos2.shape == (3,):
            pos2 = pos2[:2]
        return np.sqrt(np.sum(np.square(pos1 - pos2)))

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

    def get_goal_flag(self, robot_pos, goal_pos):
        """Get goal flat"""
        dist_goal = self.dist_xy(robot_pos, goal_pos)
        goal_size = 0.3
        return dist_goal < goal_size

    def ego_xy(self, robot_matrix, robot_pos, pos):
        '''Return the egocentric XY vector to a position from the robot'''
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = robot_pos
        robot_mat = robot_matrix

        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        robot_3vec = np.concatenate([robot_3vec, [0]])
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]

    def obs_lidar_pseudo(
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
        lidar_num_bins = 16
        lidar_max_dist = 3
        obs = np.zeros(lidar_num_bins)
        lidar_exp_gain = 1.0
        lidar_alias = True
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            position_z = np.complex(
                *self.ego_xy(robot_matrix, robot_pos, pos)
            )  # X, Y as real, imaginary components
            dist = np.abs(position_z)
            angle = np.angle(position_z) % (np.pi * 2)
            bin_size = (np.pi * 2) / lidar_num_bins
            sensor_bin = int(angle / bin_size)
            bin_angle = bin_size * sensor_bin
            if lidar_max_dist is None:
                sensor = np.exp(-lidar_exp_gain * dist)
            else:
                sensor = max(0, lidar_max_dist - dist) / lidar_max_dist
            obs[sensor_bin] = max(obs[sensor_bin], sensor)
            # Aliasing
            if lidar_alias:
                alias = (angle - bin_angle) / bin_size
                assert (
                    0 <= alias <= 1
                ), f'bad alias {alias}, dist {dist}, angle {angle}, bin {sensor_bin}'
                bin_plus = (sensor_bin + 1) % lidar_num_bins
                bin_minus = (sensor_bin - 1) % lidar_num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def make_observation(self, state, lidar):
        """Get observation"""
        state = list(state)
        lidar = list(lidar)
        base_state = state[self.key_to_slice['base_state']]
        obs = base_state + lidar + state[self.key_to_slice['robot']]

        return obs

    def generate_lidar(self, obs):
        """Get lidar observation"""
        robot_matrix_x_y = obs[self.key_to_slice['robot_m']]
        robot_matrix_x = robot_matrix_x_y[0]
        robot_matrix_y = robot_matrix_x_y[1]
        first_row = [robot_matrix_x, robot_matrix_y, 0]
        second_row = [-robot_matrix_y, robot_matrix_x, 0]
        third_row = [0, 0, 1]
        robot_matrix = [first_row, second_row, third_row]
        robot_pos = obs[self.key_to_slice['robot']]
        lidar_vec = self.obs_lidar_pseudo(robot_matrix, robot_pos, self.hazards_position)
        obs_vec = self.make_observation(obs, lidar_vec)
        return obs_vec
