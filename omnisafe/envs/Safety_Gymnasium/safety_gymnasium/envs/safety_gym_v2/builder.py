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

"""builder"""
from copy import deepcopy

import gymnasium
import gymnasium.spaces
import numpy as np
from safety_gymnasium.envs.safety_gym_v2 import tasks
from safety_gymnasium.envs.safety_gym_v2.base_task import BaseTask
from safety_gymnasium.envs.safety_gym_v2.common import ResamplingError, quat2zalign
from safety_gymnasium.envs.safety_gym_v2.engine import Engine


# from safety_gymnasium.envs.safety_gym_v2 import bases, tasks


# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256


class Builder(gymnasium.Env, gymnasium.utils.EzPickle):

    """
    Builder: an environment-building tool for safe exploration research.

    The Builder() class constructs the basic framework of environments, while the details were hidden.
    There are two important parts, one is **engine module**, which is something related to mujoco and
    general tools, the other is **task module** including all task specific operation.
    """

    # Default configuration (this should not be nested since it gets copied)
    WORLD_DEFAULT = {
        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15,
        'render_lidar_size': 0.025,
        'render_lidar_offset_init': 0.5,
        'render_lidar_offset_delta': 0.06,
        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip)
        'frameskip_binom_p': 1.0,  # Probability of trial return (controls distribution)
    }

    def __init__(self, config={}, **kwargs):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        gymnasium.utils.EzPickle.__init__(self, config=config)
        self.input_parameters = locals()
        self.get_config(config)
        self.task_id = config['task']['task_id']
        self.seed()

        self._setup_simulation()

        self.done = True

        self.render_mode = kwargs.get('render_mode', None)

    @property
    def hazards_size(self):
        """return hazards size"""
        return self.task.hazards_size

    def robot_mat(self):
        """return mat"""
        return self.task.world.robot_mat()

    def get_sensor(self, sensor):
        """return mat"""
        return self.task.world.get_sensor(sensor)

    def ego_xy(self, pos):
        """return mat"""
        return self.task.ego_xy(pos)

    @property
    def hazards_pos(self):
        """return hazards postion"""
        return self.task.hazards_pos

    @property
    def vases_pos(self):
        """return vase postion"""
        return self.task.vases_pos

    @property
    def goal_pos(self):
        """return goal position"""
        return self.task.goal_pos

    @property
    def robot_pos(self):
        '''Helper to get current robot position'''
        return self.task.robot_pos

    def get_config(self, config):
        """Parse a config dict - see self.DEFAULT for description"""
        world_config = config['world']
        task_config = config['task']

        self.world_config = deepcopy(self.WORLD_DEFAULT)
        self.world_config.update(deepcopy(world_config))

        self.task_config = deepcopy(task_config)

    def _setup_simulation(self):
        self.task = self.get_task()
        self.engine = self.get_engine(self.task)
        self.task.set_engine(self.engine)

    def get_task(
        self,
    ) -> BaseTask:
        assert hasattr(tasks, self.task_id), f'Task={self.task_id} not implemented.'
        task = getattr(tasks, self.task_id)

        return task(task_config=self.task_config)

    def get_engine(self, task):
        return Engine(task, self.world_config, self.task_config)

    def seed(self, seed=None):
        """Set internal random state seeds"""
        self._seed = np.random.randint(2**32) if seed is None else seed

    def build(self):
        """Build a new physics simulation environment"""
        self.task.build_goal()

    def set_rs(self, seed):
        rs = np.random.RandomState(seed)
        self.engine.set_rs(rs)

    def reset(self, seed=None, options=None):
        """Reset the physics simulation and return observation"""
        info = {}

        if seed is not None:
            self._seed = seed

        if not self.task.randomize_layout:
            self.set_rs(0)
        else:
            self._seed += 1  # Increment seed
            self.set_rs(self._seed)

        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)

        self.engine.reset()
        self.build()
        self.task.specific_reset()

        cost = self.cost()
        assert cost['cost'] == 0, f'World has starting cost! {cost}'

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return (self.task.obs(), info)

    def world_xy(self, pos):
        """Return the world XY vector to a position from the robot"""
        assert pos.shape == (2,)
        return pos - self.world.robot_pos()[:2]

    def reward(self):
        """Calculate the dense component of reward.  Call exactly once per step"""

        reward = self.task.calculate_reward()

        # Intrinsic reward for uprightness
        if self.task.reward_orientation:
            zalign = quat2zalign(self.data.get_body_xquat(self.reward_orientation_body))
            reward += self.reward_orientation_scale * zalign

        # Clip reward
        if self.task.reward_clip:
            in_range = reward < self.task.reward_clip and reward > -self.task.reward_clip
            if not (in_range):
                reward = np.clip(reward, -self.task.reward_clip, self.task.reward_clip)
                print('Warning: reward was outside of range!')

        return reward

    def cost(self):
        """Calculate the current costs and return a dict"""
        cost = self.task.calculate_cost()

        # Optionally remove shaping from reward functions.
        if self.task.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost

    def step(self, action):
        """Take a step and return observation, reward, done, and info"""
        assert not self.done, 'Environment must be reset before stepping'
        # action = np.array(action, copy=False)  # Cast to ndarray
        # assert not self.done, 'Environment must be reset before stepping'

        info = {}

        exception = self.engine.apply_action(action)
        if exception:
            self.done = True

            reward = self.task.reward_exception
            info['cost_exception'] = 1.0
        else:
            # Reward processing
            reward = self.reward()

            # Constraint violations
            info.update(self.cost())

            cost = info['cost']

            self.task.specific_step()
            self.task.update_world()

            # Goal processing
            if self.task.goal_achieved:
                info['goal_met'] = True
                if self.task.continue_goal:
                    # Update the internal layout so we can correctly resample (given objects have moved)
                    self.engine.update_layout()
                    # Try to build a new goal, end if we fail
                    if self.task.terminate_resample_failure:
                        try:
                            self.task.build_goal()
                        except ResamplingError as e:
                            # Normal end of episode
                            self.done = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.task.build_goal()
                else:
                    self.done = True

        # Timeout
        self.steps += 1
        if self.steps >= self.task.num_steps:
            self.done = True  # Maximum number of steps in an episode reached

        terminaled = self.done
        truncated = False

        if self.render_mode == 'human':
            self.render()
        return self.task.obs(), reward, cost, terminaled, truncated, info

    def render(self, camera_id=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        assert self.render_mode, 'Please specify the render mode when you make env.'
        assert (
            not self.task.observe_vision
        ), 'When you use vision envs, you should not call this function explicitly.'
        return self.engine.render(
            mode=self.render_mode, camera_id=camera_id, width=width, height=height, cost=self._cost
        )

    @property
    def action_space(self):
        """Helper to get action space"""
        return self.engine.action_space

    @property
    def observation_space(self):
        """Helper to get observation space"""
        return self.task.observation_space

    @property
    def obs_space_dict(self):
        """Helper to get observation space dictionary"""
        return self.engine.obs_space_dict
