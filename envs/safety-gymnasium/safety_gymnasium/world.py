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
"""World."""

import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
import safety_gymnasium
import xmltodict
import yaml
from safety_gymnasium.utils.common_utils import convert, rot2quat
from safety_gymnasium.utils.task_utils import get_body_xvelp


# Default location to look for xmls folder:
BASE_DIR = os.path.dirname(safety_gymnasium.__file__)


@dataclass
class Engine:
    """Physical engine."""

    # pylint: disable=no-member
    model: mujoco.MjModel = None
    data: mujoco.MjData = None

    def update(self, model, data):
        """Set engine."""
        self.model = model
        self.data = data


class World:  # pylint: disable=too-many-instance-attributes
    """This class starts mujoco simulation.

    And contains some apis for interacting with mujoco."""

    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Builder` configuration
    DEFAULT = {
        'agent_base': 'assets/xmls/car.xml',  # Which agent XML to use as the base
        'agent_xy': np.zeros(2),  # agent XY location
        'agent_rot': 0,  # agent rotation about Z axis
        'floor_size': [3.5, 3.5, 0.1],  # Used for displaying the floor
        # FreeGeoms -- this is processed and added by the Builder class
        'free_geoms': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},
        'floor_type': 'mat',
    }

    def __init__(self, agent, obstacles, config=None):
        """config - JSON string or dict of configuration.  See self.parse()"""
        if config:
            self.parse(config)  # Parse configuration

        self.first_reset = True

        self._agent = agent  # pylint: disable=no-member
        self._obstacles = obstacles

        self.agent_base_path = None
        self.agent_base_xml = None
        self.xml = None
        self.xml_string = None

        self.engine = Engine()
        self.bind_engine()

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description."""
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    def bind_engine(self):
        """Send the new engine instance to the agent and obstacles."""
        self._agent.set_engine(self.engine)
        for obstacle in self._obstacles:
            obstacle.set_engine(self.engine)

    def build(self):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """Build a world, including generating XML and moving objects."""
        # Read in the base XML (contains agent, camera, floor, etc)
        self.agent_base_path = os.path.join(BASE_DIR, self.agent_base)  # pylint: disable=no-member
        with open(self.agent_base_path, encoding='utf-8') as f:  # pylint: disable=invalid-name
            self.agent_base_xml = f.read()
        self.xml = xmltodict.parse(self.agent_base_xml)  # Nested OrderedDict objects

        if 'compiler' not in self.xml['mujoco']:
            compiler = xmltodict.parse(
                f'''<compiler
                angle="radian"
                meshdir="{BASE_DIR}/assets/meshes"
                texturedir="{BASE_DIR}/assets/textures"
                />'''
            )
            self.xml['mujoco']['compiler'] = compiler['compiler']
        else:
            self.xml['mujoco']['compiler'].update(
                {
                    '@angle': 'radian',
                    '@meshdir': os.path.join(BASE_DIR, 'assets', 'meshes'),
                    '@texturedir': os.path.join(BASE_DIR, 'assets', 'textures'),
                }
            )

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move agent position to starting position
        worldbody['body']['@pos'] = convert(
            # pylint: disable-next=no-member
            np.r_[self.agent_xy, self._agent.z_height]
        )
        worldbody['body']['@quat'] = convert(rot2quat(self.agent_rot))  # pylint: disable=no-member

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []
        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            self.xml['mujoco']['asset'] = {}
        if 'texture' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['texture'] = []
        if 'material' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['material'] = []
        if 'mesh' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['mesh'] = []
        material = self.xml['mujoco']['asset']['material']
        texture = self.xml['mujoco']['asset']['texture']
        mesh = self.xml['mujoco']['asset']['mesh']

        # load all assets config from .yaml file
        with open(os.path.join(BASE_DIR, 'configs/assets.yaml'), 'r', encoding='utf-8') as file:
            assets_config = yaml.load(file, Loader=yaml.FullLoader)

        texture.append(assets_config['textures']['skybox'])

        if self.floor_type == 'mat':  # pylint: disable=no-member
            texture.append(assets_config['textures']['matplane'])
            material.append(assets_config['materials']['matplane'])
        elif self.floor_type == 'village':  # pylint: disable=no-member
            texture.append(assets_config['textures']['village_floor'])
            material.append(assets_config['materials']['village_floor'])
        else:
            raise NotImplementedError

        selected_textures = {}
        selected_materials = {}
        selected_meshes = {}
        for name, config in self.geoms.items():  # pylint: disable=no-member
            if config['type'] == 'mesh':
                mesh_name = config['mesh']
                selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
        for name, config in self.free_geoms.items():  # pylint: disable=no-member
            if config['type'] == 'mesh':
                mesh_name = config['mesh']
                selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
        for name, config in self.mocaps.items():  # pylint: disable=no-member
            if config['type'] == 'mesh':
                mesh_name = config['mesh']
                selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
        texture += selected_textures.values()
        material += selected_materials.values()
        mesh += selected_meshes.values()

        # Add light to the XML dictionary
        light = xmltodict.parse(
            """<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>"""
        )
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse(
                """
                <geom name="floor" type="plane" condim="6"/>
                """
            )
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        for g in worldbody['geom']:  # pylint: disable=invalid-name
            if g['@name'] == 'floor':
                g.update(
                    {
                        '@size': convert(self.floor_size),  # pylint: disable=no-member
                        '@rgba': '1 1 1 1',
                    }
                )
                if self.floor_type == 'mat':  # pylint: disable=no-member
                    g.update({'@material': 'matplane'})
                elif self.floor_type == 'village':  # pylint: disable=no-member
                    g.update({'@material': 'village_floor'})
                else:
                    raise NotImplementedError
        # Add cameras to the XML dictionary
        cameras = xmltodict.parse(
            """<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>"""
        )
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.agent_rot  # pylint: disable=no-member
        xyaxes = dict(
            x1=np.cos(theta),
            x2=-np.sin(theta),
            x3=0,
            y1=np.sin(theta),
            y2=np.cos(theta),
            y3=1,
        )
        pos = dict(
            xp=0 * np.cos(theta) + (-2) * np.sin(theta),
            yp=0 * (-np.sin(theta)) + (-2) * np.cos(theta),
            zp=2,
        )
        track_camera = xmltodict.parse(
            """<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}"
                xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(
                **pos, **xyaxes
            )
        )
        if 'camera' in worldbody['body'][0]:
            if isinstance(worldbody['body'][0]['camera'], list):
                worldbody['body'][0]['camera'] = worldbody['body'][0]['camera'] + [
                    track_camera['b']['camera']
                ]
            else:
                worldbody['body'][0]['camera'] = [worldbody['body'][0]['camera']] + [
                    track_camera['b']['camera']
                ]
        else:
            worldbody['body'][0]['camera'] = [
                track_camera['b']['camera'],
            ]

        # Add free_geoms to the XML dictionary
        for name, object in self.free_geoms.items():  # pylint: disable=redefined-builtin, no-member
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            if name == 'push_box':
                object['quat'] = rot2quat(object['rot'])
                dim = object['size'][0]
                object['dim'] = dim
                object['width'] = dim / 2
                object['x'] = dim
                object['y'] = dim
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                    </body>
                '''.format(
                        **{k: convert(v) for k, v in object.items()}
                    )
                )
            else:
                if object['type'] == 'mesh':
                    body = xmltodict.parse(
                        # pylint: disable-next=consider-using-f-string
                        '''
                        <body name="{name}" pos="{pos}" euler="{euler}" >
                            <freejoint name="{name}"/>
                            <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}" density="{density}"
                                rgba="{rgba}" group="{group}" condim="6" />
                        </body>
                    '''.format(
                            **{k: convert(v) for k, v in object.items()}
                        )
                    )
                else:
                    object['quat'] = rot2quat(object['rot'])
                    body = xmltodict.parse(
                        # pylint: disable-next=consider-using-f-string
                        '''
                        <body name="{name}" pos="{pos}" quat="{quat}">
                            <freejoint name="{name}"/>
                            <geom name="{name}" type="{type}" size="{size}" density="{density}"
                                rgba="{rgba}" group="{group}"/>
                        </body>
                    '''.format(
                            **{k: convert(v) for k, v in object.items()}
                        )
                    )
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():  # pylint: disable=no-member
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name}'
            assert (
                name.replace('mocap', 'obj') in self.free_geoms  # pylint: disable=no-member
            ), f'missing object for {name}'  # pylint: disable=no-member
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap['rot'])
            body = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <body name="{name}" mocap="true">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        pos="{pos}" quat="{quat}" contype="0" conaffinity="0" group="{group}"/>
                </body>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()}
                )
            )
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()}
                )
            )
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():  # pylint: disable=no-member
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            if geom['type'] == 'mesh':
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" euler="{euler}">
                        <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}"
                        rgba="1 1 1 1" group="{group}" contype="{contype}"
                        conaffinity="{conaffinity}"/>
                    </body>
                '''.format(
                        **{k: convert(v) for k, v in geom.items()}
                    )
                )
            else:
                geom['quat'] = rot2quat(geom['rot'])
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        group="{group}" contype="{contype}" conaffinity="{conaffinity}"/>
                    </body>
                '''.format(
                        **{k: convert(v) for k, v in geom.items()}
                    )
                )
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)

        model = mujoco.MjModel.from_xml_string(self.xml_string)  # pylint: disable=no-member
        data = mujoco.MjData(model)  # pylint: disable=no-member

        # Recompute simulation intrinsics from new position
        mujoco.mj_forward(model, data)  # pylint: disable=no-member
        self.engine.update(model, data)

    def rebuild(self, config=None, state=True):
        """Build a new sim from a model if the model changed."""
        if state:
            old_state = self.get_state()

        if config:
            self.parse(config)
        self.build()
        if state:
            self.set_state(old_state)
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def reset(self, build=True):
        """Reset the world. (sim is accessed through self.sim)"""
        if build:
            self.build()

    def body_com(self, name):
        """Get the center of mass of a named body in the simulator world reference frame."""
        return self.data.body(name).subtree_com.copy()

    def body_pos(self, name):
        """Get the position of a named body in the simulator world reference frame."""
        return self.data.body(name).xpos.copy()

    def body_mat(self, name):
        """Get the rotation matrix of a named body in the simulator world reference frame."""
        return self.data.body(name).xmat.copy().reshape(3, -1)

    def body_vel(self, name):
        """Get the velocity of a named body in the simulator world reference frame."""
        return get_body_xvelp(self.model, self.data, name).copy()

    def get_state(self):
        """Returns a copy of the simulator state."""
        state = {}

        state['time'] = np.copy(self.data.time)
        state['qpos'] = np.copy(self.data.qpos)
        state['qvel'] = np.copy(self.data.qvel)
        if self.model.na == 0:
            state['act'] = None
        else:
            state['act'] = np.copy(self.data.act)

        return state

    def set_state(self, value):
        """
        Sets the state from an dict.

        Args:
        - value (dict): the desired state.
        - call_forward: optionally call sim.forward(). Called by default if
            the udd_callback is set.
        """
        self.data.time = value['time']
        self.data.qpos[:] = np.copy(value['qpos'])
        self.data.qvel[:] = np.copy(value['qvel'])
        if self.model.na != 0:
            self.data.act[:] = np.copy(value['act'])

    @property
    def model(self):
        """Access model easily."""
        return self.engine.model

    @property
    def data(self):
        """Access data easily."""
        return self.engine.data
