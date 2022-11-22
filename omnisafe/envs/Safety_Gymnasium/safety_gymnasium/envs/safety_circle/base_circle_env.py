import os
from typing import Optional, Union
from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space
from safety_gymnasium.envs.mujoco_env import BaseMujocoEnv

import xmltodict

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

DEFAULT_SIZE = 480

class CircleMujocoEnv(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        level = 0,
    ):
        self.model_path = model_path
        self.level = level
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f'{MUJOCO_IMPORT_ERROR}. (HINT: you need to install mujoco)'
            )
        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        with open(self.fullpath) as f:
            self.robot_base_xml = f.read()

        self.xml = xmltodict.parse(self.robot_base_xml)  # Nested OrderedDict objects
        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]

        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []
        else:
            worldbody['geom'] = []
        wall_pos = 4.8 if self.model_path.split('.')[0] == 'humanoid' else 4.8
        wall_lenth = 10 if self.model_path.split('.')[0] == 'humanoid' else 40

        circle = xmltodict.parse('''<body name="circle" pos="0 0 0.02">
                                        <geom name="circle" type="cylinder" size="10 0.01" rgba="0 1 0 0.2" contype="0" conaffinity="0" mass="0"/>
                                    </body> ''')
        worldbody['body'].append(circle['body'])

        if self.level >= 1:
            wall1 = xmltodict.parse(f'''<body name="wall1" pos="-{wall_pos} 0 0.95">
                                            <geom name="wall1" type="box" size="0.05 {wall_lenth} 1" rgba="1 0 0 0.5" contype="0" conaffinity="0" mass="0"/>
                                        </body> ''')
            worldbody['body'].append(wall1['body'])

            wall2 = xmltodict.parse(f'''<body name="wall2" pos="{wall_pos} 0 0.95">
                                            <geom name="wall2" type="box" size="0.05 {wall_lenth} 1" rgba="1 0 0 0.5" contype="0" conaffinity="0" mass="0"/>
                                        </body> ''')
            worldbody['body'].append(wall2['body'])

        if self.level >= 2:
            wall3 = xmltodict.parse(f'''<body name="wall3" pos="0 -{wall_pos} 0.95">
                                            <geom name="wall3" type="box" size="{wall_lenth} 0.05 1" rgba="1 0 0 0.5" contype="0" conaffinity="0" mass="0"/>
                                        </body> ''')
            worldbody['body'].append(wall3['body'])

            wall4 = xmltodict.parse(f'''<body name="wall4" pos="0 {wall_pos} 0.95">
                                            <geom name="wall4" type="box" size="{wall_lenth} 0.05 1" rgba="1 0 0 0.5" contype="0" conaffinity="0" mass="0"/>
                                        </body> ''')
            worldbody['body'].append(wall4['body'])

        self.xml_string = xmltodict.unparse(self.xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)

        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        if self.render_mode in {
            'rgb_array',
            'depth_array',
        }:
            camera_id = self.camera_id
            camera_name = self.camera_name

            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    'Both `camera_id` and `camera_name` cannot be' ' specified at the same time.'
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(self.render_mode).render(camera_id=camera_id)

        if self.render_mode == 'rgb_array':
            data = self._get_viewer(self.render_mode).read_pixels(depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif self.render_mode == 'depth_array':
            self._get_viewer(self.render_mode).render()
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(self.render_mode).read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif self.render_mode == 'human':
            self._get_viewer(self.render_mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()

    def _get_viewer(
        self, mode
    ) -> Union[
        'gym.envs.mujoco.mujoco_rendering.Viewer',
        'gym.envs.mujoco.mujoco_rendering.RenderContextOffscreen',
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                from gymnasium.envs.mujoco.mujoco_rendering import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {'rgb_array', 'depth_array'}:
                from gymnasium.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos
