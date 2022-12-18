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
"""Env builder."""

import gymnasium
import numpy as np
from safety_gymnasium import tasks
from safety_gymnasium.utils.common_utils import ResamplingError, quat2zalign
from safety_gymnasium.utils.task_utils import get_task_class_name


# pylint: disable-next=too-many-instance-attributes
class Builder(gymnasium.Env, gymnasium.utils.EzPickle):
    """
    Builder: an environment-building tool for safe exploration research.

    The Builder() class constructs the basic control framework of environments, while
    the details were hidden. There is another important parts, which is **task module**
    including all task specific operation.
    """

    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 125,
    }

    def __init__(self, task_id, config=None, **kwargs):
        gymnasium.utils.EzPickle.__init__(self, config=config)

        self.input_parameters = locals()

        self.task_id = task_id
        self.config = config
        self.seed()
        self._setup_simulation()

        self.first_reset = None
        self.steps = None
        self._cost = None
        self.terminated = True
        self.truncated = False

        self.render_mode = kwargs.get('render_mode', None)

    def _setup_simulation(self):
        """Set up mujoco the simulation instance."""
        self.task = self.get_task()

    def get_task(self):
        """Instantiate a task object."""
        class_name = get_task_class_name(self.task_id)
        assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
        task_class = getattr(tasks, class_name)
        task = task_class(config=self.config)

        task.build_observation_space()
        return task

    def seed(self, seed=None):
        """Set internal random state seeds."""
        self._seed = np.random.randint(2**32) if seed is None else seed

    def set_random_seed(self, seed):
        """Instantiate a :class:`np.random.RandomState` object using given seed."""
        random_generator = np.random.RandomState(seed)  # pylint: disable=no-member
        self.task.set_random_generator(random_generator)

    def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
        """Reset the physics simulation and return observation."""
        info = {}

        if seed is not None:
            self._seed = seed  # pylint: disable=attribute-defined-outside-init

        if not self.task.randomize_layout:
            self.set_random_seed(0)
        else:
            self._seed += 1  # Increment seed
            self.set_random_seed(self._seed)

        self.terminated = False
        self.truncated = False
        self.steps = 0  # Count of steps taken in this episode

        self.task.reset()
        self.task.build_goal()  # Build a new physics simulation environment
        self.task.specific_reset()
        cost = self.cost()
        assert cost['cost'] == 0, f'World has starting cost! {cost}'
        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return (self.task.obs(), info)

    def step(self, action):
        """Take a step and return observation, reward, cost, terminated, truncated, info."""
        assert not self.done, 'Environment must be reset before stepping.'
        action = np.array(action, copy=False)  # Cast to ndarray

        info = {}

        exception = self.task.apply_action(action)
        if exception:
            self.truncated = True

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
                    # Update the internal layout
                    # so we can correctly resample (given objects have moved)
                    self.task.update_layout()
                    # Try to build a new goal, end if we fail
                    if self.task.terminate_resample_failure:
                        try:
                            self.task.build_goal()
                        except ResamplingError:
                            # Normal end of episode
                            self.terminated = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.task.build_goal()
                else:
                    self.terminated = True

        # Timeout
        self.steps += 1
        if self.steps >= self.task.num_steps:
            self.truncated = True  # Maximum number of steps in an episode reached

        if self.render_mode == 'human':
            self.render()
        return self.task.obs(), reward, cost, self.terminated, self.truncated, info

    def reward(self):
        """Calculate the dense component of reward.  Call exactly once per step."""
        reward = self.task.calculate_reward()

        # Intrinsic reward for uprightness
        if self.task.reward_orientation:
            zalign = quat2zalign(
                self.task.data.get_body_xquat(self.task.reward_orientation_body)
            )  # pylint: disable=no-member
            reward += self.reward_orientation_scale * zalign  # pylint: disable=no-member

        # Clip reward
        if self.task.reward_clip:
            in_range = -self.task.reward_clip < reward < self.task.reward_clip
            if not in_range:
                reward = np.clip(reward, -self.task.reward_clip, self.task.reward_clip)
                print('Warning: reward was outside of range!')

        return reward

    def cost(self):
        """Calculate the current costs and return a dict."""
        cost = self.task.calculate_cost()

        # Optionally remove shaping from reward functions.
        if self.task.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost

    def render(self, camera_id=None, width=256, height=256):
        """Call underlying render() directly.

        Width and height in parameters are constant defaults for rendering
        frames for humans. (not used for vision)
        """
        assert self.render_mode, 'Please specify the render mode when you make env.'
        assert (
            not self.task.observe_vision
        ), 'When you use vision envs, you should not call this function explicitly.'
        return self.task.render(
            mode=self.render_mode, camera_id=camera_id, width=width, height=height, cost=self._cost
        )

    def world_xy(self, pos):
        """Return the world XY vector to a position from the robot."""
        assert pos.shape == (2,)
        return pos - self.world.robot_pos()[:2]  # pylint: disable=no-member

    @property
    def action_space(self):
        """Helper to get action space."""
        return self.task.action_space

    @property
    def observation_space(self):
        """Helper to get observation space."""
        return self.task.observation_space

    @property
    def obs_space_dict(self):
        """Helper to get observation space dictionary."""
        return self.task.obs_space_dict

    @property
    def done(self):
        """Whether this episode is ended."""
        return self.terminated or self.truncated
