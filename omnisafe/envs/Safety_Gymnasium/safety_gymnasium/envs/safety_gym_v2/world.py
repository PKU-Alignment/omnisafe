"""world"""
import os
from collections import OrderedDict
from copy import deepcopy

import mujoco
import numpy as np
import safety_gymnasium.envs.safety_gym_v2
import xmltodict
from safety_gymnasium.envs.safety_gym_v2.common import convert, rot2quat
from safety_gymnasium.envs.safety_gym_v2.robot import Robot
from safety_gymnasium.envs.safety_gym_v2.utils import get_body_xvelp

# Default location to look for /xmls folder:
BASE_DIR = os.path.dirname(safety_gymnasium.__file__)
BASE_DIR = os.path.join(BASE_DIR, 'envs', 'safety_gym_v2')


class World:
    """World: Default configuration"""

    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Builder` configuration
    DEFAULT = {
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_xy': np.zeros(2),  # Robot XY location
        'robot_rot': 0,  # Robot rotation about Z axis
        'floor_size': [3.5, 3.5, 0.1],  # Used for displaying the floor
        # Objects -- this is processed and added by the Builder class
        'objects': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},
        # Determine whether we create render contexts
        'observe_vision': False,
    }

    def __init__(self, config={}, render_context=None):
        """config - JSON string or dict of configuration.  See self.parse()"""
        self.parse(config)  # Parse configuration
        self.first_reset = True
        self.viewer = None
        self.render_context = render_context
        # self.update_viewer_sim = False
        self.robot = Robot(self.robot_base)
        self.robot_base_path = None
        self.robot_base_xml = None
        self.xml = None
        self.xml_string = None
        self.model = None
        self.data = None

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description"""
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    # TODO: remove this when mujoco-py fix is merged and a new version is pushed
    # https://github.com/openai/mujoco-py/pull/354
    # Then all uses of `self.world.get_sensor()` should change to `self.data.get_sensor`.
    def get_sensor(self, name):
        """get_sensor: Get the value of a sensor by name"""
        # pylint disable=C0103
        id = self.model.sensor(name).id
        adr = self.model.sensor_adr[id]
        dim = self.model.sensor_dim[id]
        return self.data.sensordata[adr : adr + dim].copy()

    def build(self):
        """Build a world, including generating XML and moving objects"""
        # Read in the base XML (contains robot, camera, floor, etc)
        self.robot_base_path = os.path.join(BASE_DIR, self.robot_base)
        with open(self.robot_base_path, encoding='utf-8') as f:
            self.robot_base_xml = f.read()
        self.xml = xmltodict.parse(self.robot_base_xml)  # Nested OrderedDict objects

        compiler = xmltodict.parse("""<compiler angle="radian" />""")
        self.xml['mujoco']['compiler'] = compiler['compiler']

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move robot position to starting position
        worldbody['body']['@pos'] = convert(np.r_[self.robot_xy, self.robot.z_height])
        worldbody['body']['@quat'] = convert(rot2quat(self.robot_rot))

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
            # old default rgb1: ".4 .5 .6"
            # old default rgb2: "0 0 0"
            # light pink: "1 0.44 .81"
            # light blue: "0.004 0.804 .996"
            # light purple: ".676 .547 .996"
            # med blue: "0.527 0.582 0.906"
            # indigo: "0.293 0 0.508"
            asset = xmltodict.parse(
                """
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 0.582 0.906" rgb2="0.1 0.1 0.35"
                        width="800" height="800" markrgb="1 1 1" mark="random" random="0.001"/>
                    <texture name="texplane" builtin="checker" height="100" width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>
                </asset>
                """
            )
            self.xml['mujoco']['asset'] = asset['asset']

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
        for g in worldbody['geom']:
            if g['@name'] == 'floor':
                g.update(
                    {
                        '@size': convert(self.floor_size),
                        '@rgba': '1 1 1 1',
                        '@material': 'MatPlane',
                    }
                )

        # Add cameras to the XML dictionary
        cameras = xmltodict.parse(
            """<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>"""
        )
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.robot_rot
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
            <camera name="track" mode="track" pos="{xp} {yp} {zp}" xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(
                **pos, **xyaxes
            )
        )
        worldbody['body'][0]['camera'] = [
            worldbody['body'][0]['camera'],
            track_camera['b']['camera'],
        ]

        # Add objects to the XML dictionary
        for name, object in self.objects.items():
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            object['quat'] = rot2quat(object['rot'])
            if name == 'box':
                dim = object['size'][0]
                object['dim'] = dim
                object['width'] = dim / 2
                object['x'] = dim
                object['y'] = dim
                body = xmltodict.parse(
                    """
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
                """.format(
                        **{k: convert(v) for k, v in object.items()}
                    )
                )
            else:
                body = xmltodict.parse(
                    """
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                    </body>
                """.format(
                        **{k: convert(v) for k, v in object.items()}
                    )
                )
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name}'
            assert name.replace('mocap', 'obj') in self.objects, f'missing object for {name}'
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap['rot'])
            body = xmltodict.parse(
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
                """
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()}
                )
            )
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            geom['quat'] = rot2quat(geom['rot'])
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            body = xmltodict.parse(
                """
                <body name="{name}" pos="{pos}" quat="{quat}">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}" group="{group}"
                        contype="{contype}" conaffinity="{conaffinity}"/>
                </body>
            """.format(
                    **{k: convert(v) for k, v in geom.items()}
                )
            )

            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)
        # print(self.xml_string)
        # exit(0)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)

        # Recompute simulation intrinsics from new position
        mujoco.mj_forward(self.model, self.data)

    def rebuild(self, config={}, state=True):
        """Build a new sim from a model if the model changed"""
        if state:
            old_state = self.get_state()
        # self.config.update(deepcopy(config))
        # self.parse(self.config)
        self.parse(config)
        self.build()
        if state:
            self.set_state(old_state)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, build=True):
        """Reset the world (sim is accessed through self.sim)"""
        if build:
            self.build()
        # set flag so that renderer knows to update sim
        # self.update_viewer_sim = True

    # def render(self, mode='human'):
    #     ''' Render the environment to the screen '''
    #     if self.viewer is None:
    #         self.viewer = MjViewer(self.sim)
    #         # Turn all the geom groups on
    #         self.viewer.vopt.geomgroup[:] = 1
    #         # Set camera if specified
    #         if mode == 'human':
    #             self.viewer.cam.fixedcamid = -1
    #             self.viewer.cam.type = mujoco.mjCAMERA_FREE
    #         else:
    #             self.viewer.cam.fixedcamid = self.model.camera_name2id(mode)
    #             self.viewer.cam.type = mujoco.mjCAMERA_FIXED
    #     if self.update_viewer_sim:
    #         self.viewer.update_sim(self.sim)
    #         self.update_viewer_sim = False
    #     self.viewer.render()

    def robot_com(self):
        """Get the position of the robot center of mass in the simulator world reference frame"""
        return self.body_com('robot')

    def robot_pos(self):
        """Get the position of the robot in the simulator world reference frame"""
        return self.body_pos('robot')

    def robot_mat(self):
        """Get the rotation matrix of the robot in the simulator world reference frame"""
        return self.body_mat('robot')

    def robot_vel(self):
        """Get the velocity of the robot in the simulator world reference frame"""
        return self.body_vel('robot')

    def body_com(self, name):
        """Get the center of mass of a named body in the simulator world reference frame"""
        return self.data.body(name).subtree_com.copy()

    def body_pos(self, name):
        """Get the position of a named body in the simulator world reference frame"""
        return self.data.body(name).xpos.copy()

    def body_mat(self, name):
        """Get the rotation matrix of a named body in the simulator world reference frame"""
        return self.data.body(name).xmat.copy().reshape(3, -1)

    def body_vel(self, name):
        """Get the velocity of a named body in the simulator world reference frame"""
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
