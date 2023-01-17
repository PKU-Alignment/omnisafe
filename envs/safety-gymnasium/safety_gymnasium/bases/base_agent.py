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
"""Base class for agents."""

import abc
import os
from dataclasses import dataclass, field

import glfw
import gymnasium
import mujoco
import numpy as np
import safety_gymnasium
from gymnasium import spaces
from safety_gymnasium.utils.random_generator import RandomGenerator
from safety_gymnasium.utils.task_utils import get_body_xvelp, quat2mat
from safety_gymnasium.world import Engine


BASE_DIR = os.path.dirname(safety_gymnasium.__file__)


@dataclass
class SensorConf:
    """Sensor configuration."""

    # Sensor observations
    # Specify which sensors to add to observation space
    sensors: tuple = ('accelerometer', 'velocimeter', 'gyro', 'magnetometer')
    sensors_hinge_joints: bool = True  # Observe named joint position / velocity sensors
    sensors_ball_joints: bool = True  # Observe named ball joint position / velocity sensors
    sensors_angle_components: bool = True  # Observe sin/cos theta instead of theta


@dataclass
class SensorInfo:
    """Sensor information."""

    # Needed to figure out observation space
    hinge_pos_names: list = field(default_factory=list)
    hinge_vel_names: list = field(default_factory=list)
    freejoint_pos_name: str = None
    freejoint_qvel_name: str = None
    ballquat_names: list = field(default_factory=list)
    ballangvel_names: list = field(default_factory=list)
    sensor_dim: list = field(default_factory=dict)


@dataclass
class BodyInfo:
    """Body information."""

    # Needed to figure out the observation spaces
    nq: int = None
    nv: int = None
    # Needed to figure out action space
    nu: int = None
    nbody: int = None
    # a list of geoms in the agent
    geom_names: list = field(default_factory=list)


@dataclass
class DebugInfo:
    """Debug information."""

    keys: set = field(default_factory=set)


class BaseAgent(abc.ABC):  # pylint: disable=too-many-instance-attributes
    '''Simple utility class for getting mujoco-specific info about a agent.'''

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        random_generator: RandomGenerator,
        placements: list = None,
        locations: list = None,
        keepout: float = 0.4,
        rot: float = None,
    ):
        self.placements: list = placements  # agent placements list (defaults to full extents)
        self.locations: list = (
            [] if locations is None else locations
        )  # Explicitly place agent XY coordinate
        self.keepout: float = keepout  # Needs to be set to match the agent XML used
        self.rot: float = rot  # Override agent starting angle
        self.base: str = f'assets/xmls/{name.lower()}.xml'
        self.random_generator: RandomGenerator = random_generator

        self.engine: Engine = None
        self._load_model()
        self.sensor_conf = SensorConf()
        self.sensor_info = SensorInfo()
        self.body_info = BodyInfo()
        self._init_body_info()
        self.debug_info = DebugInfo()

        # Needed to figure out z-height of free joint of offset body
        self.z_height = self.engine.data.body('agent').xpos[2]

        self.action_space = self._build_action_space()
        self._init_jnt_sensors()

    def _load_model(self):
        """Load the model from the xml file."""
        base_path = os.path.join(BASE_DIR, self.base)
        model = mujoco.MjModel.from_xml_path(base_path)  # pylint: disable=no-member
        data = mujoco.MjData(model)  # pylint: disable=no-member
        mujoco.mj_forward(model, data)  # pylint: disable=no-member
        self.set_engine(Engine(model, data))

    def _init_body_info(self):
        """Initialize body information."""
        self.body_info.nq = self.engine.model.nq
        self.body_info.nv = self.engine.model.nv
        self.body_info.nu = self.engine.model.nu
        self.body_info.nbody = self.engine.model.nbody
        self.body_info.geom_names = [
            self.engine.model.geom(i).name
            for i in range(self.engine.model.ngeom)
            if self.engine.model.geom(i).name != 'floor'
        ]

    def _build_action_space(self):
        """Build the action space for this agent."""
        bounds = self.engine.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return spaces.Box(low=low, high=high, dtype=np.float64)

    def _init_jnt_sensors(self):  # pylint: disable=too-many-branches
        """Initialize joint sensors."""
        for i in range(self.engine.model.nsensor):
            name = self.engine.model.sensor(i).name
            sensor_id = self.engine.model.sensor(
                name
            ).id  # pylint: disable=redefined-builtin, invalid-name
            self.sensor_info.sensor_dim[name] = self.engine.model.sensor(sensor_id).dim[0]
            sensor_type = self.engine.model.sensor(sensor_id).type
            if (
                # pylint: disable-next=no-member
                self.engine.model.sensor(sensor_id).objtype
                == mujoco.mjtObj.mjOBJ_JOINT  # pylint: disable=no-member
            ):  # pylint: disable=no-member
                joint_id = self.engine.model.sensor(sensor_id).objid
                joint_type = self.engine.model.jnt(joint_id).type
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:  # pylint: disable=no-member
                        self.sensor_info.hinge_pos_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL
                    ):  # pylint: disable=no-member
                        self.sensor_info.hinge_vel_names.append(name)
                    else:
                        t = self.engine.model.sensor(i).type  # pylint: disable=invalid-name
                        raise ValueError(f'Unrecognized sensor type {t} for joint')
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:  # pylint: disable=no-member
                        self.sensor_info.ballquat_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL
                    ):  # pylint: disable=no-member
                        self.sensor_info.ballangvel_names.append(name)
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:  # pylint: disable=no-member
                    # Adding slide joints is trivially easy in code,
                    # but this removes one of the good properties about our observations.
                    # (That we are invariant to relative whole-world transforms)
                    # If slide joints are added we should ensure this stays true!
                    raise ValueError('Slide joints in agents not currently supported')
            elif (
                # pylint: disable-next=no-member
                self.engine.model.sensor(sensor_id).objtype
                == mujoco.mjtObj.mjOBJ_SITE  # pylint: disable=no-member
            ):  # pylint: disable=no-member
                if name == 'agent_pos':
                    self.sensor_info.freejoint_pos_name = name
                elif name == 'agent_qvel':
                    self.sensor_info.freejoint_qvel_name = name

    def set_engine(self, engine: Engine):
        """Set the engine instance."""
        self.engine = engine

    def apply_action(self, action, noise=None):
        """Apply an action to the agent."""
        action = np.array(action, copy=False)  # Cast to ndarray

        # Set action
        action_range = self.engine.model.actuator_ctrlrange

        self.engine.data.ctrl[:] = np.clip(
            action, action_range[:, 0], action_range[:, 1]
        )  # np.clip(action * 2 / action_scale, -1, 1)

        if noise:
            self.engine.data.ctrl[:] += noise

    def build_sensor_observation_space(self):
        """Build observation space for all sensor types."""
        obs_space_dict = {}

        for sensor in self.sensor_conf.sensors:  # Explicitly listed sensors
            dim = self.sensor_info.sensor_dim[sensor]
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float64)
        # Velocities don't have wraparound effects that rotational positions do
        # Wraparounds are not kind to neural networks
        # Whereas the angle 2*pi is very close to 0, this isn't true in the network
        # In theory the network could learn this, but in practice we simplify it
        # when the sensors_angle_components switch is enabled.
        for sensor in self.sensor_info.hinge_vel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        for sensor in self.sensor_info.ballangvel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
        if self.sensor_info.freejoint_pos_name:
            sensor = self.sensor_info.freejoint_pos_name
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        if self.sensor_info.freejoint_qvel_name:
            sensor = self.sensor_info.freejoint_qvel_name
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
        # Angular positions have wraparound effects, so output something more friendly
        if self.sensor_conf.sensors_angle_components:
            # Single joints are turned into sin(x), cos(x) pairs
            # These should be easier to learn for neural networks,
            # Since for angles, small perturbations in angle give small differences in sin/cos
            for sensor in self.sensor_info.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (2,), dtype=np.float64
                )
            # Quaternions are turned into 3x3 rotation matrices
            # Quaternions have a wraparound issue in how they are normalized,
            # where the convention is to change the sign so the first element to be positive.
            # If the first element is close to 0, this can mean small differences in rotation
            # lead to large differences in value as the latter elements change sign.
            # This also means that the first element of the quaternion is not expectation zero.
            # The SO(3) rotation representation would be a good replacement here,
            # since it smoothly varies between values in all directions (the property we want),
            # but right now we have very little code to support SO(3) rotations.
            # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
            for sensor in self.sensor_info.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (3, 3), dtype=np.float64
                )
        else:
            # Otherwise include the sensor without any processing
            for sensor in self.sensor_info.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float64
                )
            for sensor in self.sensor_info.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (4,), dtype=np.float64
                )

        return obs_space_dict

    def obs_sensor(self):
        """Get observations of all sensor types."""
        obs = {}

        # Sensors which can be read directly, without processing
        for sensor in self.sensor_conf.sensors:  # Explicitly listed sensors
            obs[sensor] = self.get_sensor(sensor)
        for sensor in self.sensor_info.hinge_vel_names:
            obs[sensor] = self.get_sensor(sensor)
        for sensor in self.sensor_info.ballangvel_names:
            obs[sensor] = self.get_sensor(sensor)
        if self.sensor_info.freejoint_pos_name:
            sensor = self.sensor_info.freejoint_pos_name
            obs[sensor] = self.get_sensor(sensor)[2:]
        if self.sensor_info.freejoint_qvel_name:
            sensor = self.sensor_info.freejoint_qvel_name
            obs[sensor] = self.get_sensor(sensor)
        # Process angular position sensors
        if self.sensor_conf.sensors_angle_components:
            for sensor in self.sensor_info.hinge_pos_names:
                theta = float(self.get_sensor(sensor))  # Ensure not 1D, 1-element array
                obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
            for sensor in self.sensor_info.ballquat_names:
                quat = self.get_sensor(sensor)
                obs[sensor] = quat2mat(quat)
        else:  # Otherwise read sensors directly
            for sensor in self.sensor_info.hinge_pos_names:
                obs[sensor] = self.get_sensor(sensor)
            for sensor in self.sensor_info.ballquat_names:
                obs[sensor] = self.get_sensor(sensor)

        return obs

    def get_sensor(self, name):
        """get_sensor: Get the value of a sensor by name."""
        id = self.engine.model.sensor(name).id  # pylint: disable=redefined-builtin, invalid-name
        adr = self.engine.model.sensor_adr[id]
        dim = self.engine.model.sensor_dim[id]
        return self.engine.data.sensordata[adr : adr + dim].copy()

    def dist_xy(self, pos):
        """Return the distance from the agent to an XY position."""
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        agent_pos = self.pos
        return np.sqrt(np.sum(np.square(pos - agent_pos[:2])))

    def world_xy(self, pos):
        """Return the world XY vector to a position from the agent."""
        assert pos.shape == (2,)
        return pos - self.agent.agent_pos()[:2]  # pylint: disable=no-member

    def keyboard_control_callback(self, key, action):
        """Callback for keyboard control.

        Collect keys which are pressed.
        """
        if action == glfw.PRESS:
            self.debug_info.keys.add(key)
        elif action == glfw.RELEASE:
            self.debug_info.keys.remove(key)

    def debug(self):
        """Debug mode.

        Apply action which is inputted from keyboard.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_alive(self):
        """Returns True if the agent is alive."""

    @abc.abstractmethod
    def reset(self):
        """Called when the environment is reset."""

    @property
    def com(self):
        """Get the position of the agent center of mass in the simulator world reference frame."""
        return self.engine.data.body('agent').subtree_com.copy()

    @property
    def mat(self):
        """Get the rotation matrix of the agent in the simulator world reference frame."""
        return self.engine.data.body('agent').xmat.copy().reshape(3, -1)

    @property
    def vel(self):
        """Get the velocity of the agent in the simulator world reference frame."""
        return get_body_xvelp(self.engine.model, self.engine.data, 'agent').copy()

    @property
    def pos(self):
        """Get the position of the agent in the simulator world reference frame."""
        return self.engine.data.body('agent').xpos.copy()
