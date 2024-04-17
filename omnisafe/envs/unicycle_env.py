import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections.abc import Iterable


def to_pixel(meas_cm, shift=0):

    if isinstance(meas_cm, Iterable):
        return 1.5 * 37.795 * meas_cm + np.array(shift)

    return 1.5 * 37.795 * meas_cm + shift

class UnicycleEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, obs_config='default'):

        super(UnicycleEnv, self).__init__()

        self.dynamics_mode = 'Unicycle'
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        self.bds = np.array([[-3., -3.], [3., 3.]])

        self.dt = 0.02
        self.max_episode_steps = 1000
        self.reward_goal = 1.0
        self.goal_size = 0.3
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.initial_state = np.array([[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [-2.5, 0.0, 0.0], [2.5, -2.5, np.pi/2]])
        self.goal_pos = np.array([2.5, 2.5])
        self.rand_init = False  # Random Initial State

        self.reset()

        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()
        # Disturbance
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20

        # Build Hazards
        self.obs_config = obs_config
        self.hazards = []
        if obs_config == 'default':  # default
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([0., 0.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., 1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., -1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., -1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., 1.])})
        elif obs_config == 'test':
            # self.build_hazards(obs_config)
            self.hazards.append({'type': 'polygon', 'vertices': 0.6*np.array([[-1., -1.], [1., -1], [1., 1.], [-1., 1.]])})
            self.hazards[-1]['vertices'][:, 0] += 0.5
            self.hazards[-1]['vertices'][:, 1] -= 0.5
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., 1.])})
            self.hazards.append(
                {'type': 'polygon', 'vertices': np.array([[0.9, 0.9], [2.1, 2.1], [2.1, 0.9]])})
        else:
            n_hazards = 6
            hazard_radius = 0.6
            self.get_random_hazard_locations(n_hazards, hazard_radius)

        # Viewer
        self.viewer = None


    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        action = np.clip(action, -1.0, 1.0)
        state, reward, cost, terminated, truncated, info = self._step(action)
        return self.get_obs(), reward, cost, terminated, truncated, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        terminated : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])  #* np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()

        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist()
        reward = (self.last_goal_dist - dist_goal)  # -1e-3 * dist_goal
        self.last_goal_dist = dist_goal
        # Check if goal is met
        terminated = False
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            terminated = True
        truncated = self.episode_step >= self.max_episode_steps

        # Include constraint cost in reward (only during training, i.e. obs_config=='default')
        if self.obs_config == 'default':
            info['cost'] = 0
            for hazard in self.hazards:
                if hazard['type'] == 'circle': # They should all be circles if 'default'
                    info['cost'] += 0.1 * (np.sum((self.state[:2] - hazard['location']) ** 2) < hazard['radius'] ** 2)
        return self.state, reward, info['cost'], terminated, truncated, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self, seed=None, options=None):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state
        if self.rand_init:
            self.state = np.copy(self.initial_state[np.random.randint(self.initial_state.shape[0])])
        else:
            self.state = np.copy(self.initial_state[0])

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), dict()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step, self.state, self._goal_dist(), theta_error))

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from envs import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards)):
                if self.hazards[i]['type'] == 'circle':
                    obstacles.append(pyglet_rendering.make_circle(radius=to_pixel(self.hazards[i]['radius'], shift=0), filled=True))
                    obs_trans = pyglet_rendering.Transform(translation=(to_pixel(self.hazards[i]['location'][0], shift=screen_width/2), to_pixel(self.hazards[i]['location'][1], shift=screen_height/2)))
                    obstacles[i].set_color(1.0, 0.0, 0.0)
                    obstacles[i].add_attr(obs_trans)
                elif self.hazards[i]['type'] == 'polygon':
                    obstacles.append(pyglet_rendering.make_polygon(to_pixel(self.hazards[i]['vertices'], shift=[screen_width/2, screen_height/2]), filled=True))
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2))
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x

        return get_f, get_g

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_random_hazard_locations(self, n_hazards: int, hazard_radius: float):
        """

        Parameters
        ----------
        n_hazards : int
            Number of hazards to create
        hazard_radius : float
            Radius of hazards

        Returns
        -------
        hazards_locs : ndarray
            Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
        """

        # Create buffer with boundaries
        buffered_bds = np.copy(self.bds)
        buffered_bds[0] = buffered_bds[0] + hazard_radius
        buffered_bds[1] -= hazard_radius

        hazards = []
        hazards_centers = np.zeros((n_hazards, 2))
        n = 0  # Number of hazards actually placed
        for i in range(n_hazards):
            successfully_placed = False
            iter = 0
            hazard_type = np.random.randint(3)  # 0-> Circle 1->Square 2->Triangle
            radius = hazard_radius * (1-0.2*2.0*(np.random.random() - 0.5))
            while not successfully_placed and iter < 100:
                hazards_centers[n] = (buffered_bds[1] - buffered_bds[0]) * np.random.random(2) + buffered_bds[0]
                successfully_placed = np.all(np.linalg.norm(hazards_centers[:n] - hazards_centers[[n]], axis=1) > 3.5*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.linalg.norm(self.goal_pos - hazards_centers[n]) > 2.0*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.all(np.linalg.norm(self.initial_state[:, :2] - hazards_centers[[n]], axis=1) > 2.0*hazard_radius))
                iter += 1
            if not successfully_placed:
                continue
            if hazard_type == 0:  # Circle
                hazards.append({'type': 'circle', 'location': hazards_centers[n], 'radius': radius})
            elif hazard_type == 1:  # Square
                hazards.append({'type': 'polygon', 'vertices': np.array(
                    [[-radius, -radius], [-radius, radius], [radius, radius], [radius, -radius]])})
                hazards[-1]['vertices'] += hazards_centers[n]
            else:  # Triangle
                hazards.append({'type': 'polygon', 'vertices': np.array(
                    [[-radius, -radius], [-radius, radius], [radius, radius], [radius, -radius]])})
                # Pick a vertex and delete it
                idx = np.random.randint(4)
                hazards[-1]['vertices'] = np.delete(hazards[-1]['vertices'], idx, axis=0)
                hazards[-1]['vertices'] += hazards_centers[n]
            n += 1

        self.hazards = hazards
