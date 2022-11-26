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

"""engine"""
from copy import deepcopy
from typing import Union

import gymnasium
import gymnasium.spaces
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import RenderContextOffscreen, Viewer
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP
from safety_gymnasium.envs.safety_gym_v2.common import MujocoException, ResamplingError
from safety_gymnasium.envs.safety_gym_v2.robot import Robot
from safety_gymnasium.envs.safety_gym_v2.world import World


ORIGIN_COORDINATES = np.zeros(3)


class Engine:
    """The engine for the Safety Gymnasium environment."""

    def __init__(self, task, world_config={}, task_config={}):
        """Initialize the engine."""
        self.parse(world_config, task_config)

        self.task = task

        self.robot = Robot(self.robot_base)
        self.action_space = gymnasium.spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float64)
        self.placements = self.task.build_placements_dict()
        # self.world = self.get_world()
        self.clear()

        self.world = None
        self.viewer = None
        self._viewers = {}
        self.rs = None
        self.world_config_dict = None
        self.reset_layout = None
        self.last_action = None
        self.last_subtreecom = None
        self.layout = None
        self.observation_flatten = None

    def parse(self, world_config, task_config):
        """Parse a config dict - see self.DEFAULT for description"""
        self.world_config = {}
        self.world_config.update(deepcopy(world_config))
        for key, value in self.world_config.items():
            setattr(self, key, value)

        self.task_config = {}
        self.task_config.update(deepcopy(task_config))
        for key, value in self.task_config.items():
            setattr(self, key, value)

    def set_rs(self, rs):
        """Set the random state of the engine"""
        self.rs = rs

    def get_world(self):
        """Create a new physics simulation environment"""
        self.build_layout()
        # Build the underlying physics world
        self.world_config_dict = self.task.build_world_config(self.layout)

        assert not hasattr(self, 'world') is None, 'World exists before get_world is called.'
        world = World(self.world_config_dict)
        world.reset()
        world.build()

        return world

    def clear(self):
        """Reset internal state for building"""
        self.layout = None

    def reset(self):
        """Reset the environment."""

        self.clear()
        self.build()
        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

    def build(self):
        """Build a new physics simulation environment"""
        # Sample object positions
        self.build_layout()

        # Build the underlying physics world
        self.world_config_dict = self.task.build_world_config(self.layout)
        # print(self.task.world is self.world)
        # assert self.world is not None, 'World object does not exist.'
        if self.world is None:
            self.world = World(self.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_config_dict, state=False)
            if self.viewer:
                self.update_viewer(self.model, self.data)

        # Save last action
        self.last_action = np.zeros(self.action_space.shape)

        # Save last subtree center of mass
        self.last_subtreecom = self.world.get_sensor('subtreecom')

    def build_layout(self):
        """Rejection sample a placement of objects to find a layout."""
        for _ in range(10000):
            if self.sample_layout():
                break
        else:
            raise ResamplingError('Failed to sample layout of objects')

    def apply_action(self, action):
        """Apply an action to the robot."""
        action = np.array(action, copy=False)  # Cast to ndarray

        # Set action
        action_range = self.model.actuator_ctrlrange
        # action_scale = action_range[:,1] - action_range[:, 0]
        self.data.ctrl[:] = np.clip(
            action, action_range[:, 0], action_range[:, 1]
        )  # np.clip(action * 2 / action_scale, -1, 1)
        if self.task.action_noise:
            self.data.ctrl[:] += self.task.action_noise * self.rs.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(self.rs.binomial(self.frameskip_binom_n, self.frameskip_binom_p)):
            try:
                self.task.set_mocaps()
                mujoco.mj_step(self.model, self.data)  # Physics simulation step
            except MujocoException as me:
                print('MujocoException', me)
                exception = True
                break
        if exception:
            return exception
        else:
            mujoco.mj_forward(self.model, self.data)  # Needed to get sensor readings correct!
            return exception

    def sample_layout(self):
        """Sample a single layout, returning True if successful, else False."""

        def placement_is_valid(xy, layout):
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.task.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def draw_placement(self, placements, keepout):
        """
        Sample an (x,y) location, based on potential placement areas.

        Summary of behavior:

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify
        rectangles in the XY-plane where an object could be placed.

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y)
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally,
        randomly draw a uniform point within the selected rectangle.

        """
        if placements is None:
            choice = self.constrain_placement(self.task.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert constrained, 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.rs.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array([self.rs.uniform(xmin, xmax), self.rs.uniform(ymin, ymax)])

    def constrain_placement(self, placement, keepout):
        """Helper function to constrain a single placement by the keepout radius"""
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def update_layout(self):
        """Update layout dictionary with new places of objects"""
        mujoco.mj_forward(self.model, self.data)
        for k in list(self.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.layout[k] = self.data.body(k).xpos[:2].copy()

    def render_lidar(self, poses, color, offset, group):
        """Render the lidar observation"""
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        lidar = self.task.obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.task.lidar_type == 'pseudo':
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.task.lidar_num_bins
            rad = self.render_lidar_radius
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1, sensor + 0.1)
            self.viewer.add_marker(
                pos=pos,
                size=self.render_lidar_size * np.ones(3),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=np.array(color) * alpha,
                label='',
            )

    def render_compass(self, pose, color, offset):
        """Render a compass observation"""
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        # Truncate the compass to only visualize XY component
        compass = np.concatenate([self.obs_compass(pose)[:2] * 0.15, [offset]])
        pos = robot_pos + np.matmul(compass, robot_mat.transpose())
        self.viewer.add_marker(
            pos=pos,
            size=0.05 * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=np.array(color) * 0.5,
            label='',
        )

    def render_area(self, pos, size, color, label='', alpha=0.1):
        """Render a radial area in the environment"""
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=[size, size, z_size],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=np.array(color) * alpha,
            label=label if self.render_labels else '',
        )

    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        """Render a radial area in the environment"""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=size * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=np.array(color) * alpha,
            label=label if self.render_labels else '',
        )

    def render_swap_callback(self):
        """Callback between mujoco render and swapping GL buffers"""
        # if self.task.observe_vision and self.task.vision_render:
        #     self.viewer.draw_pixels(self.save_obs_vision, 0, 0)
        pass

    def render(self, width, height, mode, camera_id=None, camera_name=None, cost={}):
        """Render the environment to the screen"""
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        if mode in {
            'rgb_array',
            'depth_array',
        }:

            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    'Both `camera_id` and `camera_name` cannot be' ' specified at the same time.'
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "vision"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(camera_id=camera_id)

        if mode == 'human':
            self._get_viewer(mode)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        self.viewer.render_swap_callback = self.render_swap_callback
        # Turn all the geom groups on
        self.viewer.vopt.geomgroup[:] = 1
        # self.viewer.update_sim(self.sim)

        # Lidar markers
        if self.render_lidar_markers:
            offset = self.render_lidar_offset_init  # Height offset for successive lidar indicators
            if 'box_lidar' in self.obs_space_dict or 'box_compass' in self.obs_space_dict:
                if 'box_lidar' in self.obs_space_dict:
                    self.render_lidar([self.task.box_pos], COLOR['box'], offset, GROUP['box'])
                if 'box_compass' in self.obs_space_dict:
                    self.render_compass(self.task.box_pos, COLOR['box'], offset)
                offset += self.render_lidar_offset_delta
            if 'goal_lidar' in self.obs_space_dict or 'goal_compass' in self.obs_space_dict:
                if 'goal_lidar' in self.obs_space_dict:
                    self.render_lidar([self.task.goal_pos], COLOR['goal'], offset, GROUP['goal'])
                if 'goal_compass' in self.obs_space_dict:
                    self.render_compass(self.task.goal_pos, COLOR['goal'], offset)
                offset += self.render_lidar_offset_delta
            # mutiple goals
            if 'goals_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.goal_pos, COLOR['goal'], offset, GROUP['goal'])
                offset += self.render_lidar_offset_delta
            if 'apples_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.apple_pos, COLOR['apple'], offset, GROUP['apple'])
                offset += self.render_lidar_offset_delta
            if 'oranges_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.orange_pos, COLOR['orange'], offset, GROUP['orange'])
                offset += self.render_lidar_offset_delta
            if 'buttons_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.buttons_pos, COLOR['button'], offset, GROUP['button'])
                offset += self.render_lidar_offset_delta
            if 'circle_lidar' in self.obs_space_dict:
                self.render_lidar([ORIGIN_COORDINATES], COLOR['circle'], offset, GROUP['circle'])
                offset += self.render_lidar_offset_delta
            if 'walls_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.walls_pos, COLOR['wall'], offset, GROUP['wall'])
                offset += self.render_lidar_offset_delta
            if 'hazards_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.hazards_pos, COLOR['hazard'], offset, GROUP['hazard'])
                offset += self.render_lidar_offset_delta
            if 'pillars_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.pillars_pos, COLOR['pillar'], offset, GROUP['pillar'])
                offset += self.render_lidar_offset_delta
            if 'gremlins_lidar' in self.obs_space_dict:
                self.render_lidar(
                    self.task.gremlins_obj_pos,
                    COLOR['gremlin'],
                    offset,
                    GROUP['gremlin'],
                )
                offset += self.render_lidar_offset_delta
            if 'vases_lidar' in self.obs_space_dict:
                self.render_lidar(self.task.vases_pos, COLOR['vase'], offset, GROUP['vase'])
                offset += self.render_lidar_offset_delta

        # Add goal marker
        if 'ButtonTask' in self.task_id:
            self.render_area(
                self.task.goal_pos,
                self.task.buttons_size * 2,
                COLOR['button'],
                'goal',
                alpha=0.1,
            )

        # Add indicator for nonzero cost
        if cost.get('cost', 0) > 0:
            self.render_sphere(self.world.robot_pos(), 0.25, COLOR['red'], alpha=0.5)

        # Draw vision pixels
        # if self.task.observe_vision and self.task.vision_render:
        #     vision = self.obs_vision()
        #     vision = np.array(vision * 255, dtype='uint8')
        #     vision = Image.fromarray(vision).resize(self.vision_render_size)
        #     vision = np.array(vision, dtype='uint8')
        #     self.save_obs_vision = vision

        if mode == 'rgb_array':
            data = self._get_viewer(mode).read_pixels(depth=False)
            # original image is upside-down, so flip it
            self.viewer._markers[:] = []
            self.viewer._overlays.clear()
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render()
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(
        self, mode
    ) -> Union[
        'gymnasium.envs.mujoco.mujoco_rendering.Viewer',
        'gymnasium.envs.mujoco.mujoco_rendering.RenderContextOffscreen',
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = Viewer(self.model, self.data)
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            # self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def update_viewer(self, model, data):
        """update the viewer with new model and data"""
        assert self.viewer, 'Call before self.viewer existing.'
        self.viewer.model = model
        self.viewer.data = data

    def toggle_observation_space(self):
        """toggle observation space"""
        self.observation_flatten = not (self.observation_flatten)
        self.build_observation_space()

    def placements_from_location(self, location, keepout):
        """Helper to get a placements list from a given location and keepout"""
        x, y = location
        return [(x - keepout, y - keepout, x + keepout, y + keepout)]

    def build_goal_position(self):
        """Build a new goal position, maybe with resampling due to hazards"""
        # Resample until goal is compatible with layout
        if 'goal' in self.layout:
            del self.layout['goal']
        for _ in range(10000):  # Retries
            if self.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_config_dict['geoms']['goal']['pos'][:2] = self.layout['goal']
        # self.world.rebuild(deepcopy(self.world_config_dict))
        # self.update_viewer_sim = True
        goal_body_id = self.model.body('goal').id
        self.model.body(goal_body_id).pos[:2] = self.layout['goal']
        mujoco.mj_forward(self.model, self.data)

    def sample_goal_position(self):
        """Sample a new goal position and return True, else False if sample rejected"""
        placements, keepout = self.placements['goal']
        goal_xy = self.draw_placement(placements, keepout)
        for other_name, other_xy in self.layout.items():
            other_keepout = self.placements[other_name][1]
            dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
            if dist < other_keepout + self.task.placements_margin + keepout:
                return False
        self.layout['goal'] = goal_xy
        return True

    @property
    def obs_space_dict(self):
        """Return the observation space dictionary"""
        return self.task.obs_space_dict

    @property
    def model(self):
        """Helper to get the world's model instance"""
        return self.world.model

    @property
    def data(self):
        """Helper to get the world's simulation data instance"""
        return self.world.data
