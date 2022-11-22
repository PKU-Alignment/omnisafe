__credits__ = ["Rushiv Arora"]

import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from safety_gymnasium.envs.safety_circle.base_circle_env import CircleMujocoEnv
import mujoco

DEFAULT_CAMERA_CONFIG = {}


class SafetySwimmerCircleEnv(CircleMujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )
        self.level = kwargs['level']

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
            )
        CircleMujocoEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Hyperparameters
        x_pos, y_pos = xy_position_after[0], xy_position_after[1]
        xy_distance = np.linalg.norm(xy_position_after, ord=2)
        radius = 10
        x_lim = 3
        y_lim = 3

        # Get reward
        reward = (- x_velocity * y_pos + y_velocity * x_pos) / (1 + np.abs(xy_distance - radius))

        # Get cost
        cost_x = np.abs(x_pos) > x_lim
        cost_y = np.abs(y_pos) > y_lim
        cost = 0
        if self.level == 1:
            cost = float(cost_x)
        elif self.level == 2:
            cost = float(cost_x or cost_y)

        observation = self._get_obs()
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': xy_distance,

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
        }

        if cost and self.viewer:
            self.add_cost_marker(pos=self.data.qpos[0:2].copy())

        if self.render_mode == 'human':
            self.render()
        return observation, reward, cost, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
                
    def add_cost_marker(self, pos):
        pos = np.hstack((pos, np.array(1)))
        color = [0.5, 0 ,0, 0.5]

        self.viewer.add_marker(pos=pos,
                        size=.4 * np.ones(3),
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        rgba=color,
                        label='')