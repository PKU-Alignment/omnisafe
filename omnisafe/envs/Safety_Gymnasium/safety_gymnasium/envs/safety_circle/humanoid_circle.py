import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from safety_gymnasium.envs.safety_circle.base_circle_env import CircleMujocoEnv
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class SafetyHumanoidCircleEnv(CircleMujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )
        self.level = kwargs['level']

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        CircleMujocoEnv.__init__(
            self, "humanoid.xml", 5, observation_space=observation_space, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()


        com_inertia = self.data.cinert.flat[:140].copy()
        com_velocity = self.data.cvel.flat[:84].copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat[:84].copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Hyperparameters
        x_pos, y_pos = xy_position_after[0], xy_position_after[1]
        xy_distance = np.linalg.norm(xy_position_after, ord=2)
        radius = 10
        x_lim = 2.5
        y_lim = 2.5

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
        terminated = self.terminated
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': xy_distance,

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
        }

        if cost and self.viewer:
            self.add_cost_marker(pos=self.get_body_com('torso')[:3].copy())

        if self.render_mode == 'human':
            self.render()
        return observation, reward, cost, terminated, False, info

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
        pos = pos + np.array([0, 0, 0.1])

        color = [0.5, 0 ,0, 0.5]

        self.viewer.add_marker(pos=pos,
                        size=.4 * np.ones(3),
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        rgba=color,
                        label='')