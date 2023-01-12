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
"""Swimmer environment with a safety constraint on velocity."""

import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer


class SafetySwimmerVelocityEnv(SwimmerEnv):
    """Swimmer environment with a safety constraint on velocity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._velocity_threshold = 0.04845
        self.model.light(0).castshadow = False

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            'reward_fwd': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),
            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        cost = velocity > self._velocity_threshold

        if self.viewer:
            clear_viewer(self.viewer)
            add_velocity_marker(
                viewer=self.viewer,
                pos=self.get_body_com('torso')[:3].copy(),
                vel=velocity,
                cost=cost,
                velocity_threshold=self._velocity_threshold,
            )
        if self.render_mode == 'human':
            self.render()
        return observation, reward, cost, False, False, info
