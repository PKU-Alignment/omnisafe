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

import numpy as np


def dist_xy(pos1, pos2):
    '''Return the distance from the robot to an XY position'''
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    if pos1.shape == (3,):
        pos1 = pos1[:2]
    if pos2.shape == (3,):
        pos2 = pos2[:2]
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


# -------Assuming we have reward & cost function. available with us in closed form.
def get_reward_cost(last_dist_goal, robot_pos, hazards_pos, goal_pos):
    # ------cost hazard-----------
    cost = 0
    hazards_size = 0.2
    hazards_cost = 1.0
    for h_pos in hazards_pos:
        h_dist = dist_xy(h_pos, robot_pos)
        if h_dist <= hazards_size:
            cost += hazards_cost * (hazards_size - h_dist)
    if cost > 0:
        cost = 1
    else:
        cost = 0
    # ----reward-----------------

    reward = 0
    reward_distance = 1.0
    reward_goal = 1.0
    goal_size = 0.3

    dist_goal = dist_xy(robot_pos, goal_pos)
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
    return reward, cost, last_dist_goal, goal_flag


def get_reward_cost_v2(last_dist_goal, current_dist_goal, hazards_pos):
    # ------cost hazard-----------
    cost = 0
    hazards_size = 0.2
    hazards_cost = 1.0
    for i in range(0, 18, 2):
        hazard_pos = hazards_pos[i : i + 2]
        hazard_dis = np.sqrt(np.sum(np.square(hazard_pos)))
        if hazard_dis <= hazards_size:
            cost += hazards_cost * (hazards_size - hazard_dis)
    if cost > 0:
        cost = 1
    else:
        cost = 0
    # ----reward-----------------

    reward = 0
    reward_distance = 1.0
    reward_goal = 1.0
    goal_size = 0.3

    dist_goal = current_dist_goal
    reward += (last_dist_goal - current_dist_goal) * reward_distance
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
    return reward, cost, last_dist_goal, goal_flag


def get_goal_flag(robot_pos, goal_pos):
    dist_goal = dist_xy(robot_pos, goal_pos)
    goal_size = 0.3
    if dist_goal < goal_size:
        return True
    else:
        return False


def ego_xy(robot_matrix, robot_pos, pos):
    '''Return the egocentric XY vector to a position from the robot'''
    assert pos.shape == (2,), f'Bad pos {pos}'
    robot_3vec = robot_pos
    robot_mat = robot_matrix

    pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
    robot_3vec = np.concatenate([robot_3vec, [0]])
    world_3vec = pos_3vec - robot_3vec
    return np.matmul(world_3vec, robot_mat)[:2]


# def obs_lidar(positions):
#     '''
#     Calculate and return a lidar observation.  See sub methods for implementation.
#     '''
#     return obs_lidar_pseudo(positions)
def obs_lidar_pseudo(robot_matrix, robot_pos, positions):
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
        z = np.complex(*ego_xy(robot_matrix, robot_pos, pos))  # X, Y as real, imaginary components
        dist = np.abs(z)
        angle = np.angle(z) % (np.pi * 2)
        bin_size = (np.pi * 2) / lidar_num_bins
        bin = int(angle / bin_size)
        bin_angle = bin_size * bin
        if lidar_max_dist is None:
            sensor = np.exp(-lidar_exp_gain * dist)
        else:
            sensor = max(0, lidar_max_dist - dist) / lidar_max_dist
        obs[bin] = max(obs[bin], sensor)
        # Aliasing
        if lidar_alias:
            alias = (angle - bin_angle) / bin_size
            assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
            bin_plus = (bin + 1) % lidar_num_bins
            bin_minus = (bin - 1) % lidar_num_bins
            obs[bin_plus] = max(obs[bin_plus], alias * sensor)
            obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
    return obs


def make_observation(state, lidar):
    state = list(state)
    lidar = list(lidar)
    x = state[:8]
    obs = x + lidar + state[40:]
    return obs


def generate_lidar(o, hazards_pos):
    robot_matrix_x_y = o[38:40]
    x = robot_matrix_x_y[0]
    y = robot_matrix_x_y[1]

    first_row = [x, y, 0]
    second_row = [-y, x, 0]
    third_row = [0, 0, 1]
    robot_matrix = [first_row, second_row, third_row]
    robot_pos = o[40:]
    # --------------------------------------------------------------------------------------
    lidar_vec = obs_lidar_pseudo(robot_matrix, robot_pos, hazards_pos)
    obs_vec = make_observation(o, lidar_vec)
    return obs_vec
