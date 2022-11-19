from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.button import get_button
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP
from safety_gymnasium.envs.safety_gym_v2.base_task import BaseTask


class ButtonLevel0(BaseTask):
    """A task"""

    def __init__(
        self,
        task_config,
    ):
        super().__init__(
            task_config=task_config,
        )
        self.placements_extents = [-1, -1, 1, 1]

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        self.reward_distance = 1.0  # Dense reward multiplied by the distance moved to the goal

        self.buttons_num = 4
        self.buttons_size = 0.1
        self.buttons_keepout = 0.2
        self.observe_buttons = True

        self.hazards_size = 0.2
        self.hazards_keepout = 0.18

        self.gremlins_travel = 0.35
        self.gremlins_keepout = 0.4

        self.agent_specific_config()
        self.last_dist_goal = None
        self.buttons_timer = None
        self.goal_button = None

    # pylint: disable=W0613
    def calculate_cost(self, **kwargs):
        """determine costs depending on agent and obstacles"""
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost

    def calculate_reward(self):
        """Returns the reward of an agent running in a circle (clock-wise)."""
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.reward_goal

        return reward

    @property
    def goal_achieved(self):
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if any(n == f'button{self.goal_button}' for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    return True
        return False

    @property
    def goal_pos(self):
        """Helper to get goal position from layout"""
        return self.data.body(f'button{self.goal_button}').xpos.copy()

    @property
    def buttons_pos(self):
        """Helper to get the list of button positions"""
        return [self.data.body(f'button{i}').xpos.copy() for i in range(self.buttons_num)]

    @property
    def hazards_pos(self):
        """Helper to get the hazards positions from layout"""
        return [self.data.body(f'hazard{i}').xpos.copy() for i in range(self.hazards_num)]

    @property
    def gremlins_obj_pos(self):
        """Helper to get the current gremlin position"""
        return [self.data.body(f'gremlin{i}obj').xpos.copy() for i in range(self.gremlins_num)]

    def agent_specific_config(self):
        pass

    def specific_reset(self):
        """Reset agent position and set orientation towards desired run
        direction."""
        self.buttons_timer = 0

    def specific_step(self):
        self.buttons_timer_tick()

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards"""
        assert self.buttons_num > 0, 'Must have at least one button'
        self.build_goal_button()
        self.last_dist_goal = self.dist_goal()
        self.buttons_timer = self.buttons_resampling_delay

    def build_placements_dict(self):
        """Build a dict of placements.  Happens once during __init__."""
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))

        placements.update(self.placements_dict_from_object('button'))

        return placements

    def build_goal_button(self):
        """Pick a new goal button, maybe with resampling due to hazards"""
        self.goal_button = self.rs.choice(self.buttons_num)

    def build_world_config(self, layout):
        """Create a world_config from our own config"""
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = layout['robot']
        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        # if self.floor_display_mode:
        #     floor_size = max(self.placements_extents)
        #     world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        # if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        # world_config['observe_vision'] = self.observe_vision

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        # if self.buttons_num:
        for i in range(self.buttons_num):
            name = f'button{i}'
            world_config['geoms'][name] = get_button(
                index=i, layout=layout, rot=self.random_rot(), size=self.buttons_size
            )

        return world_config

    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__"""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.build_sensor_observation_space())

        # if self.observe_goal_lidar:
        obs_space_dict['goal_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32
        )
        # if self.buttons_num and self.observe_buttons:
        obs_space_dict['buttons_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32
        )

        # if self.observe_vision:
        #     width, height = self.vision_size
        #     rows, cols = height, width
        #     self.vision_size = (rows, cols)
        #     obs_space_dict['vision'] = gym.spaces.Box(0, 1.0, self.vision_size + (3,), dtype=np.float32)

        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        # if self.observation_flatten:
        self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float64
        )
        # else:
        #     self.observation_space = gym.spaces.Dict(obs_space_dict)

    def obs(self):
        """Return the observation of our agent"""
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensordata correct
        obs = {}

        # if self.observe_goal_lidar:
        obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP['goal'])

        obs.update(self.get_sensor_obs())

        # if self.buttons_num and self.observe_buttons:
        # Buttons observation is zero while buttons are resetting
        if self.buttons_timer == 0:
            obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP['button'])
        else:
            obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)

        # if self.observe_vision:
        #     obs['vision'] = self.obs_vision()

        # if self.observation_flatten:
        flat_obs = np.zeros(self.obs_flat_size)
        offset = 0
        for k in sorted(self.obs_space_dict.keys()):
            k_size = np.prod(obs[k].shape)
            flat_obs[offset : offset + k_size] = obs[k].flat
            offset += k_size
        obs = flat_obs
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs

    def buttons_timer_tick(self):
        """Tick the buttons resampling timer"""
        #  Button timer (used to delay button resampling)
        self.buttons_timer = max(0, self.buttons_timer - 1)

    def set_mocaps(self, gremlin_size=0.1):
        """Set mocap object positions before a physics step is executed"""
        phase = float(self.data.time)
        for i in range(self.gremlins_num):
            name = f'gremlin{i}'
            target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins_travel
            pos = np.r_[target, [gremlin_size]]
            self.set_mocap_pos(name + 'mocap', pos)

    def set_mocap_pos(self, name, value):
        """Set the position of a mocap object"""
        body_id = self.model.body(name).id
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = value
