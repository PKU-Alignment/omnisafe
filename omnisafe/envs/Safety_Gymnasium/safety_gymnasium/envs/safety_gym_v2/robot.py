"""Robot"""
import os

import mujoco
import safety_gymnasium.envs.safety_gym_v2


BASE_DIR = os.path.dirname(safety_gymnasium.__file__)
BASE_DIR = os.path.join(BASE_DIR, 'envs', 'safety_gym_v2')


class Robot:
    """Simple utility class for getting mujoco-specific info about a robot"""

    def __init__(self, path):
        base_path = os.path.join(BASE_DIR, path)
        self.model = mujoco.MjModel.from_xml_path(base_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Needed to figure out z-height of free joint of offset body
        self.z_height = self.data.body('robot').xpos[2]
        # Get a list of geoms in the robot
        self.geom_names = [
            self.model.geom(i).name
            for i in range(self.model.ngeom)
            if self.model.geom(i).name != 'floor'
        ]

        # Needed to figure out the observation spaces
        self.nq = self.model.nq
        self.nv = self.model.nv
        # Needed to figure out action space
        self.nu = self.model.nu
        # Needed to figure out observation space
        # See engine.py for an explanation for why we treat these separately
        self.hinge_pos_names = []
        self.hinge_vel_names = []
        self.ballquat_names = []
        self.ballangvel_names = []
        self.sensor_dim = {}
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            id = self.model.sensor(name).id
            self.sensor_dim[name] = self.model.sensor(id).dim[0]
            sensor_type = self.model.sensor(id).type
            if self.model.sensor(id).objtype == mujoco.mjtObj.mjOBJ_JOINT:
                joint_id = self.model.sensor(id).objid
                joint_type = self.model.jnt(joint_id).type
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:
                        self.hinge_pos_names.append(name)
                    elif sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL:
                        self.hinge_vel_names.append(name)
                    else:
                        t = self.model.sensor(i).type
                        raise ValueError(f'Unrecognized sensor type {t} for joint')
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:
                        self.ballquat_names.append(name)
                    elif sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL:
                        self.ballangvel_names.append(name)
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    # Adding slide joints is trivially easy in code,
                    # but this removes one of the good properties about our observations.
                    # (That we are invariant to relative whole-world transforms)
                    # If slide joints are added we sould ensure this stays true!
                    raise ValueError('Slide joints in robots not currently supported')
