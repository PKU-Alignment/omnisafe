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

import gymnasium
import numpy as np
import safety_gymnasium

from omnisafe.algorithms.model_based.aux import (
    dist_xy,
    ego_xy,
    generate_lidar,
    get_goal_flag,
    get_reward_cost,
    make_observation,
    obs_lidar_pseudo,
)


# ----------------------------------------------------------------------------------------------------------
ROBOTS = ['Point', 'Car', 'Doggo']
TASKS = ['Goal', 'Button']

XYZ_SENSORS = dict(
    Point=['velocimeter'],
    Car=[
        'velocimeter'
    ],  # ,'accelerometer'],#,'ballquat_rear', 'right_wheel_vel', 'left_wheel_vel'],
    Doggo=['velocimeter', 'accelerometer'],
)

ANGLE_SENSORS = dict(
    Point=['gyro', 'magnetometer'], Car=['magnetometer', 'gyro'], Doggo=['magnetometer', 'gyro']
)

CONSTRAINTS = dict(
    Goal=['vases', 'hazards'],
    # Goal=['hazards'],
    Button=['hazards', 'gremlins', 'buttons'],
)
CONSTRAINTS_MBPPO = dict(
    Goal=['hazards'],
    Button=['hazards', 'gremlins', 'buttons'],
)


class EnvWrapper:
    def __init__(self, algo, env_id, render_mode='none'):
        self.algo = algo
        self.env_id = env_id  # safety gym not use this attribute
        if self.algo == 'MBPPOLag':
            assert env_id == 'SafetyPointGoal3-v0' or env_id == 'SafetyCarGoal3-v0'
            self.robot = 'Point' if 'Point' in env_id else 'Car'
            self.task = 'Goal'
        elif self.algo == 'mbppo_v2':
            assert env_id == 'SafetyPointGoal1-v0' or env_id == 'SafetyCarGoal1-v0'
            self.robot = 'Point' if 'Point' in env_id else 'Car'
            self.task = 'Goal'
        elif self.algo == 'SafeLoop':
            assert env_id == 'SafetyPointGoal1-v0' or env_id == 'SafetyCarGoal1-v0'
            self.robot = 'Point' if 'Point' in env_id else 'Car'
            self.task = 'Goal'

        elif self.algo == 'CAP' or self.algo == 'MpcCcem':
            assert env_id == 'HalfCheetah-v3'

        mujoco_pools = [
            'Ant-v3',
            'Swimmer-v3',
            'HalfCheetah-v3',
            'Hopper-v3',
            'Humanoid-v3',
            'Walker2d-v3',
        ]
        self.env_type = 'mujoco' if self.env_id in mujoco_pools else 'gym'
        if self.env_type == 'gym':
            self.robot = self.robot.capitalize()  # mujuco  not use this attribute
            self.task = self.task.capitalize()  # mujuco  not use this attribute
            assert self.robot in ROBOTS, 'can not recognize the robot type {}'.format(self.robot)
            assert self.task in TASKS, 'can not recognize the task type {}'.format(self.task)
            env_name = self.env_id
            self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
            self.init_sensor()
            self.env.num_steps = 1000
            self.num_steps = 1000
            # for uses with ppo in baseline
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32
            )
            self.action_space = gymnasium.spaces.Box(
                -1, 1, (self.env.action_space.shape[0],), dtype=np.float32
            )

        else:  # mujoco
            env_name = self.env_id

            self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

    def get_observation_cost(self, obs):
        N = obs.shape[0]
        hazards_pos_list = self.env.hazards_pos  # list of shape (3,) ndarray
        ego_hazards_pos_list = [
            self.env.ego_xy(pos[:2]) for pos in hazards_pos_list
        ]  # list of shape (2,) ndarray

        hazards_key = self.key_to_slice['hazards']

        hazard_obs = obs[:, hazards_key].reshape(N, -1, 2)
        hazards_dist = np.sqrt(np.sum(np.square(hazard_obs), axis=2)).reshape(N, -1)
        cost = (
            (hazards_dist < self.env.hazards_size) * (self.env.hazards_size - hazards_dist)
        ).sum(1) * 10

        return cost

    def init_sensor(self):
        self.xyz_sensors = XYZ_SENSORS[self.robot]
        self.angle_sensors = ANGLE_SENSORS[self.robot]
        self.constraints_name = CONSTRAINTS[self.task]
        self.constraints_mbppo = CONSTRAINTS_MBPPO[self.task]
        # self.distance_name = ["goal_dist"] + [x+"_dist" for x in self.constraints_name]

        self.base_state_name = self.xyz_sensors + self.angle_sensors
        if self.algo == 'MBPPOLag':
            self.flatten_order = (
                self.base_state_name + ['goal'] + self.constraints_mbppo + ['robot_m'] + ['robot']
            )  # + self.distance_name
        elif self.algo == 'SafeLoop' or self.algo == 'mbppo_v2':
            self.flatten_order = (
                self.base_state_name + ['goal'] + self.constraints_name
            )  # + self.distance_name

        # get state space vector size
        self.env.reset()
        obs = self.get_obs()
        self.obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        self.state_dim = self.obs_flat_size
        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)
            # print("obs key: ", k, " slice: ", self.key_to_slice[k])
            offset += k_size
        self.base_state_dim = sum([np.prod(obs[k].shape) for k in self.base_state_name])
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice['base_state'] = slice(0, self.base_state_dim)

    def reset(self):
        if self.env_type == 'mujoco':
            obs = self.env.reset()
            return obs

        else:
            self.t = 0  # Reset internal timer
            self.env.reset()
            obs = self.get_obs_flatten()
            if self.algo == 'MBPPOLag':
                hazards_pos_list = self.env.hazards_pos
                goal_pos = self.env.goal_pos
                static_detail = {'hazards': hazards_pos_list, 'goal': goal_pos}
                return obs, static_detail
            elif self.algo == 'SafeLoop' or self.algo == 'mbppo_v2':
                return obs

    def step(self, action, num_repeat):
        # 2 dimensional numpy array, [vx, w]

        if self.env_type == 'mujoco':
            next_o, r, d, info = self.env.step(action)
            if 'y_velocity' not in info:
                c = np.abs(info['x_velocity'])
            else:
                c = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)

            return next_o, r, c, d, info
        else:
            reward = 0
            cost = 0

            for k in range(num_repeat):
                control = action
                state, reward_k, cost, terminated, truncated, info = self.env.step(control)
                reward += reward_k
                cost += cost
                self.t += 1  # Increment internal timer
                observation = self.get_obs_flatten()
                goal_met = 'goal_met' in info.keys()  # reach the goal
                done = terminated or truncated
                if terminated or goal_met or truncated:
                    break
            if self.algo == 'MBPPOLag':
                cost = 1 if cost > 0 else 0
                info = {'cost': cost, 'goal_met': goal_met, 'goal_pos': self.env.goal_pos}
            elif self.algo == 'SafeLoop' or self.algo == 'mbppo_v2':
                info = {'cost': cost, 'goal_met': goal_met}

            return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def recenter(self, pos):
        '''Return the egocentric XY vector to a position from the robot'''
        return self.env.ego_xy(pos)

    def dist_xy(self, pos):
        '''Return the distance from the robot to an XY position, 3 dim or 2 dim'''
        return self.env.dist_xy(pos)

    def get_obs(self):
        '''
        We will ignore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos  # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos  # list of shape (3,) ndarray
        # gremlins_pos_list = self.env.gremlins_obj_pos  # list of shape (3,) ndarray
        # buttons_pos_list = self.env.buttons_pos  # list of shape (3,) ndarray

        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = [
            self.env.ego_xy(pos[:2]) for pos in vases_pos_list
        ]  # list of shape (2,) ndarray
        ego_hazards_pos_list = [
            self.env.ego_xy(pos[:2]) for pos in hazards_pos_list
        ]  # list of shape (2,) ndarray
        # ego_gremlins_pos_list = [
        #     self.env.ego_xy(pos[:2]) for pos in gremlins_pos_list
        # ]  # list of shape (2,) ndarray
        # ego_buttons_pos_list = [
        #     self.env.ego_xy(pos[:2]) for pos in buttons_pos_list
        # ]  # list of shape (2,) ndarray

        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor == 'accelerometer':
                obs[sensor] = self.env.get_sensor(sensor)[:1]  # only x axis matters
            elif sensor == 'ballquat_rear':
                obs[sensor] = self.env.get_sensor(sensor)
            else:
                obs[sensor] = self.env.get_sensor(sensor)[:2]  # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.get_sensor(sensor)[2:]  # [2:] # only z axis matters
                # pass # gyro does not help
            else:
                obs[sensor] = self.env.get_sensor(sensor)
        if self.algo == 'MBPPOLag':
            # --------modification-----------------
            obs['robot'] = np.array(robot_pos[:2])
            # obs["static_goal"] = np.array(goal_pos)
            # obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
            obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
            robot_matrix = self.env.robot_mat()
            obs['robot_m'] = np.array(robot_matrix[0][:2])
            obs['goal'] = ego_goal_pos  # (2,)
            # obs["gremlins"] = np.array(ego_gremlins_pos_list) # (vase_num, 2)
            # obs["buttons"] = np.array(ego_buttons_pos_list) # (hazard_num, 2)
        elif self.algo == 'SafeLoop' or self.algo == 'mbppo_v2':
            obs['vases'] = np.array(ego_vases_pos_list)  # (vase_num, 2)
            obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
            obs['goal'] = ego_goal_pos  # (2,)
            # obs["gremlins"] = np.array(ego_gremlins_pos_list)  # (vase_num, 2)
            # obs["buttons"] = np.array(ego_buttons_pos_list)  # (hazard_num, 2)

        return obs

    def get_obs_flatten(self):
        # get the flattened obs
        self.obs = self.get_obs()
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = self.obs[k].flat
        return flat_obs

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.dist_goal()

    @property
    def observation_size(self):
        return self.state_dim

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self.env.action_space.low[0]), float(self.env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self.env.action_space.sample()
