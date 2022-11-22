__credits__ = ["Rushiv Arora"]

import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from safety_gymnasium.envs.mujoco_env import MujocoEnv
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SafetyHalfCheetahVelocityEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
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
        self._velocity_threshold = 2

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self, "half_cheetah.xml", 5, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        cost = x_velocity > self._velocity_threshold
        
        if self.viewer:
            self.add_velocity_marker(pos=self.get_body_com('torso')[:3].copy(), vel=x_velocity, cost=cost)

        if self.render_mode == 'human':
            self.render()
        return observation, reward, cost, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
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

    def add_velocity_marker(self, pos, vel, cost):

        pos = pos + np.array([0, 0, 0.6])
        safe_color = np.array([0.2, 0.8 ,0.2, 0.5])
        unsafe_color = np.array([0.5, 0, 0, 0.5])

        if cost:
            color = unsafe_color
        else:
            vel_ratio = vel / self._velocity_threshold
            color = safe_color * (1 - vel_ratio)

        self.viewer.add_marker(pos=pos,
                        size=.2 * np.ones(3),
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        rgba=color,
                        label='')