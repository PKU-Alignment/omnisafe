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
"""Robot."""

import os
from dataclasses import InitVar, dataclass, field

import mujoco
import safety_gymnasium


BASE_DIR = os.path.dirname(safety_gymnasium.__file__)


@dataclass
class Robot:  # pylint: disable=too-many-instance-attributes
    '''Simple utility class for getting mujoco-specific info about a robot.'''

    path: InitVar[str]

    placements: list = None  # Robot placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Explicitly place robot XY coordinate
    keepout: float = 0.4  # Needs to be set to match the robot XML used
    base: str = 'assets/xmls/car.xml'  # Which robot XML to use as the base
    rot: float = None  # Override robot starting angle

    def __post_init__(self, path):
        self.base = path
        base_path = os.path.join(BASE_DIR, path)
        self.model = mujoco.MjModel.from_xml_path(base_path)  # pylint: disable=no-member
        self.data = mujoco.MjData(self.model)  # pylint: disable=no-member
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

        # Needed to figure out z-height of free joint of offset body
        self.z_height = self.data.body('robot').xpos[2]
        # Get a list of geoms in the robot
        self.geom_names = [
            self.model.geom(i).name
            for i in range(self.model.ngeom)
            if self.model.geom(i).name != 'floor'
        ]

        # Needed to figure out the observation spaces
        self.nq = self.model.nq  # pylint: disable=invalid-name
        self.nv = self.model.nv  # pylint: disable=invalid-name
        # Needed to figure out action space
        self.nu = self.model.nu  # pylint: disable=invalid-name
        # Needed to figure out observation space
        # See engine.py for an explanation for why we treat these separately
        self.hinge_pos_names = []
        self.hinge_vel_names = []
        self.ballquat_names = []
        self.ballangvel_names = []
        self.sensor_dim = {}
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            id = self.model.sensor(name).id  # pylint: disable=redefined-builtin, invalid-name
            self.sensor_dim[name] = self.model.sensor(id).dim[0]
            sensor_type = self.model.sensor(id).type
            if (
                # pylint: disable-next=no-member
                self.model.sensor(id).objtype
                == mujoco.mjtObj.mjOBJ_JOINT  # pylint: disable=no-member
            ):  # pylint: disable=no-member
                joint_id = self.model.sensor(id).objid
                joint_type = self.model.jnt(joint_id).type
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:  # pylint: disable=no-member
                        self.hinge_pos_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL
                    ):  # pylint: disable=no-member
                        self.hinge_vel_names.append(name)
                    else:
                        t = self.model.sensor(i).type  # pylint: disable=invalid-name
                        raise ValueError(f'Unrecognized sensor type {t} for joint')
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:  # pylint: disable=no-member
                        self.ballquat_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL
                    ):  # pylint: disable=no-member
                        self.ballangvel_names.append(name)
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:  # pylint: disable=no-member
                    # Adding slide joints is trivially easy in code,
                    # but this removes one of the good properties about our observations.
                    # (That we are invariant to relative whole-world transforms)
                    # If slide joints are added we should ensure this stays true!
                    raise ValueError('Slide joints in robots not currently supported')
