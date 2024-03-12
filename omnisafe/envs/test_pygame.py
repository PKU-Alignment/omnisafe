from typing import Any, ClassVar, Tuple, Optional, Union, List
import gym
import numpy as np
import pygame as pg
from gym.core import ActType, ObsType, RenderFrame
import sys
import random

from typing import Any, ClassVar, Tuple, Optional, Union, List

import gym
import numpy as np
import pygame as pg
import torch
from gym.core import ActType, ObsType, RenderFrame
import random
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box


class DrawCircle(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2,
        'render_fps': 1000,
        'render_mode': 'human'
    }
    env_config = {
        'control_type': 'velocity',
        'random_reset': False,
        'constraints': np.array([0.4, 0.6]),
        'pos_max_min': (-1.0, 1.0),
        'vel_max_min': (-1.0, 1.0),
        'time_diff': 0.1
    }

    def __init__(self, **kwargs):
        self.action_space = gym.spaces.Box(low=np.array([self.env_config['vel_max_min'][0]] * 2),
                                           high=np.array([self.env_config['vel_max_min'][1]] * 2),
                                           dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([self.env_config['pos_max_min'][0]] * 4),
                                                high=np.array([self.env_config['pos_max_min'][1]] * 4),
                                                dtype=np.float64)

        for key, value in kwargs.items():
            self.env_config[key] = value
        self.time_diff = self.env_config['time_diff']
        pg.init()
        size = (500, 500)
        self.screen_size = size
        if self.metadata['render_mode'] == 'human':
            self.render_screen = pg.display.set_mode(self.screen_size)
        self.screen = pg.Surface(size)
        self._max_episode_step = 1000

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.screen = pg.Surface(self.screen_size)
        self.step_count = 0
        if self.env_config['random_reset']:
            random_reset = self.observation_space.sample()
            random_reset[2:] = self.action_space.sample()
        else:
            radiums = np.random.uniform(0.1, 1)
            theta = np.random.uniform(0, 2 * np.pi)
            x = radiums * np.cos(theta)
            y = radiums * np.sin(theta)
            self.freq = np.random.uniform(0, 1)
            v_x = -radiums * self.freq * np.cos(theta)
            v_y = radiums * self.freq * np.sin(theta)
            random_reset = np.array([x, y, v_x, v_y])
        self.item_pos = random_reset[:2]
        self.item_vel = random_reset[2:]
        self.centre_pos = np.zeros(2)
        self.item_accel = np.zeros(2)
        self.trajectory = np.zeros([self._max_episode_step, self.observation_space.shape[0]])
        return random_reset, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        if action.ndim > 1:
            action = action.squeeze(0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        done = False
        if self.env_config['control_type'] == 'velocity':
            self.item_vel = action
            self.item_pos += self.item_vel * self.time_diff
            self.item_pos = np.clip(self.item_pos, self.observation_space.low[:2], self.observation_space.high[:2])

        elif self.env_config['control_type'] == 'accelerate':
            self.item_accel = action

            vel_0 = self.item_vel
            vel_1 = vel_0 + action * self.time_diff
            vel_1 = np.clip(vel_1, self.observation_space.low[2:], self.observation_space.high[2:])
            vel_ave = (vel_0 + vel_1) / 2
            self.item_vel = vel_1
            self.item_pos += vel_ave * self.time_diff
            self.item_pos = np.clip(self.item_pos, self.observation_space.low[:2], self.observation_space.high[:2])

        obs = np.concatenate([self.item_pos, self.item_vel])
        self.trajectory[self.step_count] = obs[:]
        self.step_count += 1
        reward = self.step_reward()
        cost = self.step_cost()
        if self.step_count >= self._max_episode_step:
            done = True
        return obs, reward, cost, done, False, {}

    def render(self, mode='gif') -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        background_color = (255, 255, 255)
        traj_color = (0, 0, 0)
        centre_color = (0, 0, 255)
        item_color = (255, 0, 0)
        coordinate_color = (0, 255, 0)
        forbidden_color = (64, 64, 0)
        # out_range_color = (0, 64, 64)

        self.screen.fill(background_color)
        # draw_out_range
        circle_center = self.transpos2render(self.centre_pos)
        self.screen.fill(forbidden_color)
        out_radius = self.env_config['constraints'][1] * self.screen_size[0] // 2

        pg.draw.circle(self.screen, background_color, circle_center, out_radius, 0)

        # draw_in_range
        in_radius = self.env_config['constraints'][0] * self.screen_size[0] // 2
        pg.draw.circle(self.screen, forbidden_color, circle_center, in_radius, 0)

        # draw_coordinate
        axis_c_pos = self.transpos2render((0, 0))
        x_axis_start = self.transpos2render((-1, 0))
        x_axis_end = self.transpos2render((1, 0))
        y_axis_end = self.transpos2render((0, 1))
        y_axis_start = self.transpos2render((0, -1))
        pg.draw.circle(self.screen, coordinate_color, axis_c_pos, 1)
        pg.draw.aaline(self.screen, coordinate_color, x_axis_start, x_axis_end)
        pg.draw.aaline(self.screen, coordinate_color, y_axis_start, y_axis_end)
        # draw_centre
        centre_pos = self.transpos2render(self.centre_pos)
        pg.draw.circle(self.screen, centre_color, centre_pos, 3)
        # draw_item
        item_pos = self.transpos2render(self.item_pos)
        pg.draw.circle(self.screen, item_color, item_pos, 3)

        mid_pos = np.array([self.screen_size[0] // 2, self.screen_size[1] // 2])
        trajectory_render = mid_pos + np.array([1, -1]) * self.trajectory[:self.step_count, :2] * mid_pos
        line_point_list = trajectory_render.tolist()
        if len(line_point_list) > 1:
            pg.draw.aalines(self.screen, traj_color, False, line_point_list)

        if mode == 'human':
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
            self.render_screen.blit(self.screen, self.screen.get_rect())
            pg.display.update()

        img = pg.surfarray.array3d(self.screen)
        # self.gif_buffer.append(img)
        # print(self.step_count)
        # if self.step_count == self._max_episode_step - 1:
        #     images = [PIL.Image.fromarray(img.astype('uint8'), 'RGB') for img in self.gif_buffer]
        #     # 使用Pillow库的save()函数将图像列表保存为GIF动画
        #     images[0].save('animation.gif', save_all=True, append_images=images[1:], optimize=False,
        #                    duration=5 / self._max_episode_step,
        #                    loop=0)
        return img

    def step_reward(self):
        # mass_centre = self.trajectory[:self.step_count].mean(0)
        # distance = (self.trajectory - self.centre_pos)[:self.step_count]
        # distance = (distance[:, 0] ** 2 + distance[:, 1] ** 2) ** 0.5
        # circle_shape_reward = distance.std()
        #
        # mc_dis = mass_centre - self.centre_pos
        # circle_centre_reward = (mc_dis[0] ** 2 + mc_dis[1] ** 2) ** 0.5

        # distance = self.item_pos - self.centre_pos
        # distance = (distance[0] ** 2 + distance[1] ** 2) ** 0.5
        # min_range = self.env_config['constraints'][0]
        # max_range = self.env_config['constraints'][1]
        # if min_range < distance < max_range:
        #     range_reward = 1
        # else:
        #     range_reward = -1

        speed_reward = (self.item_vel[0] ** 2 + self.item_vel[1] ** 2) ** 0.5

        return speed_reward

    def step_cost(self):
        item_distance = self.item_pos - self.centre_pos
        item_distance = (item_distance[0] ** 2 + item_distance[1] ** 2) ** 0.5
        if item_distance < self.env_config['constraints'][1] and item_distance > self.env_config['constraints'][0]:
            return 0
        else:
            return 1

    def episode_reward(self, done):
        if done:
            distance = (self.trajectory[:, :2] - self.centre_pos)[:]
            distance = (distance[:, 0] ** 2 + distance[:, 1] ** 2) ** 0.5
            min_range = self.env_config['constraints'][0]
            max_range = self.env_config['constraints'][1]
            sat = np.where(np.where((distance > min_range) & (distance < max_range)))
            constraints_reward = len(sat[0]) / len(distance)
            return constraints_reward
        else:
            return 0

    def get_constraints(self):
        item_distance = self.item_pos - self.centre_pos
        item_distance = (item_distance[0] ** 2 + item_distance[1] ** 2) ** 0.5
        if item_distance > self.env_config['constraints'][1]:
            constraint = np.array([1.0, 0.0])
        elif item_distance < self.env_config['constraints'][0]:
            constraint = np.array([0.0, 1.0])
        else:
            constraint = random.sample(
                [np.array([1.0, 0.0]), np.array([0.0, 1.0])], 1)[0]
        return constraint

    def transpos2render(self, pos):

        midx = self.screen_size[0] // 2
        midy = self.screen_size[1] // 2
        x = pos[0]
        y = pos[1]
        render_x = int(midx + x * midx)
        render_y = int(midy - y * midy)
        return (render_x, render_y)


def collect_expert_trajectory(traj_num=1000, type='circle', save_path='./data'):
    def action_sample(step, radiums, theta, freq, type, time_diff):
        time = time_diff * step
        if type == 'circle':
            action = np.array(
                [-radiums * freq * np.sin(freq * time + theta),
                 radiums * freq * np.cos(freq * time + theta)])
        elif type == 'rect':
            edge = radiums / 2 ** 0.5
            freqL = 4 * edge / np.pi * freq
            vel_change_time = int(2 * edge / (freqL * time_diff))
            # if step % vel_change_time == 0:
            current_theta = theta + 0.75 * np.pi + (step // vel_change_time) * np.pi * 0.5
            action = np.array(
                [freqL * np.cos(current_theta),
                 freqL * np.sin(current_theta)])

        return action

    env = DrawCircle()
    env.env_config['random_reset'] = False
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    data_dim = obs_dim * 2 + act_dim + 1 + 1 + 1 + 1 + 2 + 2
    dataset = np.zeros((traj_num * env._max_episode_step, data_dim), dtype=np.float32)
    total_step = 0
    for i in range(traj_num):
        obs, info = env.reset()
        type = random.sample(['circle', 'rect'], 1)[0]
        done = False
        radiums = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
        theta = np.arcsin(obs[1] / radiums)
        if obs[0] < 0 and obs[1] > 0:
            theta = np.pi - theta
        if obs[0] < 0 and obs[1] < 0:
            theta = -np.pi - theta

        step = 0
        while not done:

            action = action_sample(step=step, radiums=radiums, theta=theta, freq=env.freq, type=type,
                                   time_diff=env.time_diff)
            next_obs, reward, cost, done, tructe, _ = env.step(action)
            constraint = env.get_constraints()
            if type == 'circle':
                skill = np.array([1.0, 0.0])
            elif type == 'rect':
                skill = np.array([0.0, 1.0])
            dataset[total_step] = np.concatenate(
                [
                    obs, action, next_obs, np.array([reward, cost, done, tructe]), constraint, skill
                ]
            )
            step += 1
            total_step += 1
            obs = next_obs
            env.render(mode='human')

    obs = dataset[:, :obs_dim]
    action = dataset[:, obs_dim:obs_dim + act_dim]
    next_obs = dataset[:, obs_dim + act_dim:2 * obs_dim + act_dim]
    reward = dataset[:, 2 * obs_dim + act_dim:2 * obs_dim + act_dim + 1]
    cost = dataset[:, 2 * obs_dim + act_dim + 1:2 * obs_dim + act_dim + 2]
    done = dataset[:, 2 * obs_dim + act_dim + 2:2 * obs_dim + act_dim + 3]
    constraint = dataset[:, 2 * obs_dim + act_dim + 4:2 * obs_dim + act_dim + 6]
    skill = dataset[:, 2 * obs_dim + act_dim + 6:2 * obs_dim + act_dim + 8]
    save_path = save_path + '/' + 'SafetyDrawCircle-v0_data_expert'
    np.savez(save_path, obs=obs, action=action, next_obs=next_obs, reward=reward, cost=cost, done=done,
             constraint=constraint, skill=skill)


if __name__ == '__main__':
    collect_expert_trajectory(traj_num=2000, type='circle',
                              save_path='C:/Users\zhou2\.cache\omnisafe\datasets')
