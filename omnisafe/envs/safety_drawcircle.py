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
"""SafeDrawCircle environment."""

import random
from typing import Any, ClassVar, List, Optional, Tuple, Union

import gym
import numpy as np
import pygame as pg
import torch
from gym.core import ActType, ObsType, RenderFrame

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box


# pylint: disable=W,C,R,E
class DrawCircle(gym.Env):
    """Implementation of gym environment."""

    metadata: ClassVar[dict[str]] = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2,
        'render_fps': 1000,
        'render_mode': 'gif',
    }
    env_config: ClassVar[dict[str]] = {
        'control_type': 'velocity',
        'random_reset': False,
        'constraints': np.array([0.4, 0.6]),
        'pos_max_min': (-1.0, 1.0),
        'vel_max_min': (-1.0, 1.0),
        'time_diff': 0.02,
        'max_episode_step': 1000,
    }

    def __init__(self, **kwargs: dict) -> None:
        """Initialize an instance of Class DrawCircle."""
        self.action_space = gym.spaces.Box(
            low=np.array([self.env_config['vel_max_min'][0]] * 2),
            high=np.array([self.env_config['vel_max_min'][1]] * 2),
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([self.env_config['pos_max_min'][0]] * 4),
            high=np.array([self.env_config['pos_max_min'][1]] * 4),
            dtype=np.float64,
        )

        for key, value in kwargs.items():
            self.env_config[key] = value
        self.time_diff = self.env_config['time_diff']
        pg.init()
        size = (500, 500)
        self.screen_size = size
        if self.metadata['render_mode'] == 'human':
            self.render_screen = pg.display.set_mode(self.screen_size)
        self.screen = pg.Surface(size)
        self._max_episode_step = self.env_config['max_episode_step']

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Reset Environment."""
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
            random_reset = np.array([x, y, 0.0, 0.0])
        self.item_pos = random_reset[:2]
        self.item_vel = random_reset[2:]
        self.centre_pos = np.zeros(2)
        self.item_accel = np.zeros(2)
        self.trajectory = np.zeros([self._max_episode_step, self.observation_space.shape[0]])
        return random_reset, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Step function of environment."""
        if action.ndim > 1:
            action = action.squeeze(0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        done = False
        if self.env_config['control_type'] == 'velocity':
            self.item_vel = action
            self.item_pos += self.item_vel * self.time_diff
            self.item_pos = np.clip(
                self.item_pos,
                self.observation_space.low[:2],
                self.observation_space.high[:2],
            )

        elif self.env_config['control_type'] == 'accelerate':
            self.item_accel = action

            vel_0 = self.item_vel
            vel_1 = vel_0 + action * self.time_diff
            vel_1 = np.clip(vel_1, self.observation_space.low[2:], self.observation_space.high[2:])
            vel_ave = (vel_0 + vel_1) / 2
            self.item_vel = vel_1
            self.item_pos += vel_ave * self.time_diff
            self.item_pos = np.clip(
                self.item_pos,
                self.observation_space.low[:2],
                self.observation_space.high[:2],
            )

        obs = np.concatenate([self.item_pos, self.item_vel])
        self.trajectory[self.step_count] = obs[:]
        self.step_count += 1
        reward = self.step_reward()
        cost = self.step_cost()
        if self.step_count >= self._max_episode_step:
            done = True
            self.reset()
        return obs, reward, cost, done, False, {}

    def render(self, mode: str = 'gif') -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Virtualization for environment."""
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
        trajectory_render = (
            mid_pos + np.array([1, -1]) * self.trajectory[: self.step_count, :2] * mid_pos
        )
        line_point_list = trajectory_render.tolist()
        if len(line_point_list) > 1:
            pg.draw.aalines(self.screen, traj_color, False, line_point_list)

        # if self.metadata['render_mode'] == 'human':
        #     for event in pg.event.get():
        #         if event.type == pg.QUIT:
        #             sys.exit()
        #     self.render_screen.blit(self.screen, self.screen.get_rect())
        #     pg.display.update()

        return pg.surfarray.array3d(self.screen)
        # self.gif_buffer.append(img)
        # print(self.step_count)
        # if self.step_count == self._max_episode_step - 1:
        #     images = [PIL.Image.fromarray(img.astype('uint8'), 'RGB') for img in self.gif_buffer]
        #     images[0].save('animation.gif', save_all=True, append_images=images[1:], optimize=False,
        #                    duration=5 / self._max_episode_step,
        #                    loop=0)

    def step_reward(self) -> np.ndarray:
        """Calculate step reward for environment transition."""
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

        return (self.item_vel[0] ** 2 + self.item_vel[1] ** 2) ** 0.5

    def step_cost(self) -> int:
        """Calculate step cost for environment transition."""
        item_distance = self.item_pos - self.centre_pos
        item_distance = (item_distance[0] ** 2 + item_distance[1] ** 2) ** 0.5
        if (
            item_distance < self.env_config['constraints'][1]
            and item_distance > self.env_config['constraints'][0]
        ):
            return 0

        return 1

    def episode_reward(self, done: bool) -> float:
        """Calculate reward when an episode is done."""
        if done:
            distance = (self.trajectory[:, :2] - self.centre_pos)[:]
            distance = (distance[:, 0] ** 2 + distance[:, 1] ** 2) ** 0.5
            min_range = self.env_config['constraints'][0]
            max_range = self.env_config['constraints'][1]
            sat = np.where(np.where((distance > min_range) & (distance < max_range)))
            return len(sat[0]) / len(distance)

        return 0

    def get_constraints(self) -> np.ndarray:
        """Get constraints for decision diffuser input."""
        item_distance = self.item_pos - self.centre_pos
        item_distance = (item_distance[0] ** 2 + item_distance[1] ** 2) ** 0.5
        if item_distance > self.env_config['constraints'][1]:
            constraint = np.array([1.0, 0.0])
        elif item_distance < self.env_config['constraints'][0]:
            constraint = np.array([0.0, 1.0])
        else:
            constraint = random.sample([np.array([1.0, 0.0]), np.array([0.0, 1.0])], 1)[0]
        return constraint

    def transpos2render(self, pos: tuple) -> tuple:
        """Virtualization function."""
        midx = self.screen_size[0] // 2
        midy = self.screen_size[1] // 2
        x = pos[0]
        y = pos[1]
        render_x = int(midx + x * midx)
        render_y = int(midy - y * midy)
        return (render_x, render_y)


@env_register
class SafeDrawCircle(CMDP):
    """Implementation of CDMP environment."""

    # _action_space: OmnisafeSpace
    # _observation_space: OmnisafeSpace
    # _metadata: dict[str, Any]

    _num_envs: int = 1
    _time_limit: int | None = None
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = ['SafetyDrawCircle-v0']

    @classmethod
    def support_envs(cls) -> list[str]:
        """The supported environments.

        Returns:
            The supported environments.
        """
        return cls._support_envs

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`CMDP`."""
        assert (
            env_id in self.support_envs()
        ), f'env_id {env_id} is not supported by {self.__class__.__name__}'
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._env = DrawCircle(**kwargs)
        self._action_space = Box(
            high=self._env.action_space.high,
            low=self._env.action_space.low,
            shape=self._env.action_space.shape,
            dtype=self._env.action_space.dtype,
        )
        self._observation_space = Box(
            high=self._env.observation_space.high,
            low=self._env.observation_space.low,
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
        )
        self._metadata = self._env.metadata

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for this env's random number generator(s).

        Args:
            seed (int): The seed to use.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the action space.

        Returns:
            The sampled action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components.
        """
        return {}

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
