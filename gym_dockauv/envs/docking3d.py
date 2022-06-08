import datetime
import importlib
import logging
import os
import pprint
import time
from abc import abstractmethod
from timeit import default_timer as timer
from typing import Tuple, Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.objects.current import Current
from gym_dockauv.objects.sensor import Radar
from gym_dockauv.objects.auvsim import AUVSim
from gym_dockauv.objects.shape import Sphere, Spheres, Capsule, intersec_dist_line_capsule_vectorized, \
    intersec_dist_lines_spheres_vectorized, collision_sphere_spheres, collision_capsule_sphere
import gym_dockauv.objects.shape as shape
import gym_dockauv.utils.geomutils as geom

# TODO: Add logging, which subclass has been called

# Set logger
logger = logging.getLogger(__name__)


class BaseDocking3d(gym.Env):
    """
    Base Class for the docking environment, will also be registered with gym. However, the configs for the
    environment are found at gym_dockauv/config

    .. note:: Adding a reward or a done condition with reward needs to take the following steps:
        - Add reward to the self.last_reward_arr in the reward step
        - Add a factor to it in the config file
        - Update number of self.n_rewards in __init__()
        - Update the list self.meta_data_reward in __init__()
        - Update the index of self.meta_data_done in __init__() if necessary
        - Update the doc for the reward_step() function (and of done())

    .. note:: Adding observations:
        - Add plus one to self.n_observation in __init__, update meta data here too
        - Add observation in observe() function, clip it accordingly, maybe update self.observation_space
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__()
        # Basic config for logger
        self.config = env_config
        self.title = self.config["title"]
        self.save_path_folder = self.config["save_path_folder"]
        self.log_level = self.config["log_level"]
        self.verbose = self.config["verbose"]
        self.interval_episode_log = self.config["interval_episode_log"]
        os.makedirs(self.save_path_folder, exist_ok=True)
        # Initialize logger
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        logging.basicConfig(level=self.log_level,
                            filename=os.path.join(self.save_path_folder, f"{utc_str}__{self.title}.log"),
                            format='[%(asctime)s] [%(levelname)s] [%(module)s] - [%(funcName)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
        # Check if logging statement are supposed to go to std output
        if self.verbose:
            logging.getLogger().addHandler(logging.StreamHandler())
        logging.Formatter.converter = time.gmtime  # Make sure to use UTC time in logging timestamps

        logger.info('---------- Docking3d Gym Logger ----------')
        logger.info('---------- ' + utc_str + ' ----------')
        logger.info('---------- Initialize environment ----------')
        logger.info('Gym environment settings: \n ' + pprint.pformat(env_config))

        # Dynamically load class of vehicle and instantiate it (available vehicles under gym_dockauv/objects/vehicles)
        AUV = getattr(importlib.import_module("gym_dockauv.objects.vehicles." + self.config["vehicle"]),
                      self.config["vehicle"])
        self.auv = AUV()  # type: AUVSim

        # Set step size for vehicle
        self.auv.step_size = self.config["t_step_size"]

        # Set assumption values for vehicle max velocities from config
        self.u_max = self.config["u_max"]
        self.v_max = self.config["v_max"]
        self.w_max = self.config["w_max"]
        self.p_max = self.config["p_max"]
        self.q_max = self.config["q_max"]
        self.r_max = self.config["r_max"]

        # Navigation errors
        self.delta_d = 0
        self.delta_psi = 0
        self.delta_theta = 0
        self.delta_heading_goal = 0  # This is the error for the heading that is required AT goal

        # Water current
        self.current = Current(mu=0.005, V_min=0.0, V_max=0.0, Vc_init=0.0,
                               alpha_init=np.pi / 4, beta_init=np.pi / 4, white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)

        # Init radar sensor suite
        self.radar_args = self.config["radar"]
        self.radar = Radar(eta=self.auv.eta, **self.radar_args)

        # Init list of obstacles (that will collide with the vehicle or have intersection with the radar)
        # Keep copy of capsules and spheres, as they are the only one supported so far:
        self.capsules = []  # type: list[Capsule]
        self.spheres = Spheres([])  # type: Spheres
        self.obstacles = [*self.capsules, *self.spheres()]  # type: list[shape.Shape]

        # Set the action and observation space
        self.n_obs_without_radar = 16
        self.n_observations = self.n_obs_without_radar + self.radar.n_rays
        self.action_space = gym.spaces.Box(low=self.auv.u_bound[:, 0],
                                           high=self.auv.u_bound[:, 1],
                                           dtype=np.float32)
        # Except for delta distance and rays, observation is between -1 and 1
        obs_low = -np.ones(self.n_observations)
        obs_low[0] = 0
        obs_low[self.n_obs_without_radar:] = 0
        self.observation_space = gym.spaces.Box(low=obs_low,
                                                high=np.ones(self.n_observations),
                                                dtype=np.float32)
        self.observation = np.zeros(self.n_observations)
        # The inner lists decide, in which subplot the observations will go
        self.meta_data_observation = [
            ["delta_d", "delta_theta", "delta_psi"],
            ["u", "v", "w"],
            ["phi", "theta", "psi_sin", "psi_cos"],
            ["p", "q", "r"],
            ["u_c", "v_c", "w_c"],
            [f"ray_{i}" for i in range(self.radar.n_rays)]
        ]

        # General simulation variables:
        self.t_total_steps = 0  # Number of total timesteps run so far in this environment
        self.t_steps = 0  # Number of steps in this episode
        self.t_step_size = self.config["t_step_size"]
        self.episode = 0  # Current episode
        self.max_timesteps = self.config["max_timesteps"]
        self.interval_datastorage = self.config["interval_datastorage"]
        self.info = {}  # This will contain general simulation info

        # Declaring further own attributes
        self.goal_reached = False  # Bool to check of goal is reached at the end of an episode
        self.collision = False  # Bool to indicate of vehicle has collided

        # Rewards
        self.n_rewards = 9
        self.last_reward = 0  # Last reward
        self.last_reward_arr = np.zeros(self.n_rewards)  # This should reflect the dimension of the rewards parts
        self.cumulative_reward = 0  # Current cumulative reward of agent
        self.cum_reward_arr = np.zeros(self.n_rewards)
        self.conditions = None  # Boolean array to see which conditions are true
        # Description for the meta data
        self.meta_data_reward = [
            "Nav_errors",
            "Attitude",
            "time_step",
            "action",
            "Done-Goal_reached",
            "Done-out_pos",
            "Done-out_att",
            "Done-max_t",
            "Done-collision"
        ]
        self.reward_factors = self.config["reward_factors"]  # Dictionary containing weights!
        # Extraxt done rewards in an array
        self.w_done = np.array([
            self.reward_factors["w_goal"],
            self.reward_factors["w_deltad_max"],
            self.reward_factors["w_Theta_max"],
            self.reward_factors["w_t_max"],
            self.reward_factors["w_col"]
        ])
        self.action_reward_factors = self.config["action_reward_factors"]

        # Initialize Done condition and related stuff for the done condition
        self.done = False
        self.meta_data_done = self.meta_data_reward[4:]
        self.goal_constraints = []  # List of booleans for further contraints as soon as goal is reached
        self.goal_location = None  # This needs to be defined in self.generate_environment
        self.dist_goal_reached_tol = self.config[
            "dist_goal_reached_tol"]  # Distance tolerance for successfully reached goal
        self.velocity_goal_reached_tol = self.config["velocity_goal_reached_tol"]  # Velocity limit at goal
        self.ang_rate_goal_reached_tol = self.config["ang_rate_goal_reached_tol"]  # Angular rate limit at goal
        self.attitude_goal_reached_tol = self.config["attitude_goal_reached_tol"]  # Attitude tolerance at goal
        self.max_dist_from_goal = self.config["max_dist_from_goal"]
        self.max_attitude = self.config["max_attitude"]
        self.heading_goal_reached = 0  # Heading at goal, pitch should be zero

        # Save and display simulation time
        self.start_time_sim = timer()

        # Episode Data storage
        self.episode_data_storage = None

        # Full data storage:
        self.full_data_storage = FullDataStorage()
        self.full_data_storage.set_up_episode_storage(env=self, path_folder=self.save_path_folder, title=self.title)

        # Animation variables
        self.episode_animation = None
        self.ax = None

        logger.info('---------- Initialization of environment complete ---------- \n')
        logger.info('---------- Rewards function description ----------')
        logger.info(self.reward_step.__doc__)

    def reset(self, seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,
              ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        From Base Class:

        Resets the environment to an initial state and returns an initial
        observation.

        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and `reset` is called with `seed=None`,
        the RNG should not be reset.
        Moreover, `reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        .. note:: Options parameter not used yet
        """
        # In case any windows were open from matplotlib or animation
        if self.episode_animation:
            plt.close(self.episode_animation.fig)  # TODO: Window stays open, prob due to Gym
            self.episode_animation = None
            self.ax = None

        # Save info to return in the end
        return_info_dict = self.info.copy()

        # Check if we should save a datastorage item
        if self.episode_data_storage and (self.episode % self.interval_datastorage == 0 or self.episode == 1):
            self.episode_data_storage.save()
        self.episode_data_storage = None

        # Update Full data storage:
        if self.episode != 0:
            self.full_data_storage.update()

        # ---------- General reset from here on -----------
        self.auv.reset()
        self.t_steps = 0
        self.goal_reached = False
        self.collision = False
        self.info = {}

        # Reset observation, cum_reward, done, info
        self.observation = np.zeros(self.n_observations, dtype=np.float32)
        self.last_reward = 0
        self.cumulative_reward = 0
        self.last_reward_arr = np.zeros(self.n_rewards)
        self.cum_reward_arr = np.zeros(self.n_rewards)
        self.done = False
        self.conditions = None
        self.goal_constraints = []

        # Water current reset
        self.current = Current(mu=0.005, V_min=0.0, V_max=0.0, Vc_init=0.0,
                               alpha_init=np.pi / 4, beta_init=np.pi / 4, white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)

        # Radar reset:
        self.radar.reset(self.auv.eta)

        # Obstacles reset
        self.obstacles = []

        # Navigation errors
        self.delta_d = 0
        self.delta_psi = 0
        self.delta_theta = 0

        # Update the seed:
        # TODO: Check if this makes all seeds same (e.g. for water current!!) or works in general
        # Comment Thomas: maybe need to fix at 2-3 other places
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            np.random.seed(seed)

        # Log the episode
        if self.episode == 1 or self.episode % self.interval_episode_log == 0:
            logger.info("Environment reset call: \n" + pprint.pformat(return_info_dict))
        else:
            logger.debug("Environment reset call: \n" + pprint.pformat(return_info_dict))

        # Update episode number
        self.episode += 1

        # ----------- Init for new environment from here on -----------
        # Generate environment
        self.generate_environment()

        # Init whole episode data if interval is met, or we need it for the renderer
        if self.episode % self.interval_datastorage == 0 or self.episode == 1:
            self.init_episode_storage()
        else:
            self.episode_data_storage = None

        # Return info if wanted
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    @abstractmethod
    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios,
        order matters in some helper functions

        """
        # Goal location:
        self.goal_location = None
        # Derive desired heading attitude at goal:
        self.heading_goal_reached = None
        # Position
        self.auv.position = None
        # Attitude
        self.auv.attitude = None
        # Water current
        self.current = None
        self.nu_c = None  # self.current(self.auv.attitude)
        # Obstacles:
        self.obstacles = None
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Simulate and update current in body frame
        self.current.sim()
        self.nu_c = self.current(self.auv.attitude)

        # Update AUV dynamics
        self.auv.step(action, self.current(self.auv.attitude))

        # Update radar
        self.radar.update(self.auv.eta)
        i_dist = self.update_radar_collision()
        self.radar.update_intersec(intersec_dist=i_dist)

        # Check collision between AUV and obstacles
        self.collision = self.update_body_collision()

        # Update data storage if active
        if self.episode_data_storage:
            self.episode_data_storage.update(self.nu_c)

        # Update visualization if active
        if self.episode_animation:
            self.render()

        # Update navigation errors
        self.update_navigation_errors()  # Update navigation errors

        # Make next observation
        self.observation = self.observe()

        # Determine if simulation is done, this also updates self.last_reward
        self.done, cond_idx = self.is_done()

        # Calculate rewards
        self.last_reward = self.reward_step(action)
        self.cumulative_reward += self.last_reward

        # Save sim time info
        self.t_total_steps += 1
        self.t_steps += 1

        # Update info dict
        self.info = {"episode_number": self.episode,  # Need to be episode number, because episode is used by sb3
                     "t_step": self.t_steps,
                     "t_total_steps": self.t_total_steps,
                     "cumulative_reward": self.cumulative_reward,
                     "last_reward": self.last_reward,
                     "done": self.done,
                     "conditions_true": cond_idx,
                     "conditions_true_info": [self.meta_data_done[i] for i in cond_idx],
                     "collision": self.collision,
                     "goal_reached": self.goal_reached,
                     "goal_contraints": self.goal_constraints,
                     "simulation_time": timer() - self.start_time_sim}

        return self.observation, self.last_reward, self.done, self.info

    def update_navigation_errors(self):
        """
        Update some navigation error values and save them to instance
        :return:
        """
        diff = self.goal_location - self.auv.position
        self.delta_d = np.linalg.norm(diff)
        self.delta_theta = self.auv.attitude[1] + (geom.ssa(np.arctan2(diff[2], np.linalg.norm(diff[:2]))))
        self.delta_psi = geom.ssa(np.arctan2(diff[1], diff[0]) - self.auv.attitude[2])
        self.delta_heading_goal = self.heading_goal_reached - self.auv.attitude[2]

    def update_radar_collision(self) -> Union[np.ndarray, None]:
        """
        Function to update the radar collision, MUST be called after radar position and attitude update to reflect
        recent radar vectors

        :return: array(nr, 1) number of rays nr with the closest collision point in direction of each radar
        """
        i_dist_list = []
        # Calculate with capsules
        for capsule in self.capsules:
            i_dist_list.append(
                intersec_dist_line_capsule_vectorized(
                    l1=self.radar.pos_arr, ld=self.radar.rd_n, cap1=capsule.vec_bot, cap2=capsule.vec_top,
                    cap_rad=capsule.radius)
            )
        # Then calculate intersection with spheres
        if len(self.spheres()) > 0:
            i_dist_list.append(
                intersec_dist_lines_spheres_vectorized(
                    l1=self.radar.pos_arr, ld=self.radar.rd_n, center=self.spheres.position, rad=self.spheres.radius)
            )
        # Get the smaller positive value of all intersection (as this will be the first intersection point)
        if len(i_dist_list) > 0:
            i_dist = np.vstack([*i_dist_list]).T  # Unfortunately slow, but good expandable solution
            i_dist = i_dist[np.arange(i_dist.shape[0]), np.where(i_dist > 0, i_dist, np.inf).argmin(axis=1)]
        else:
            i_dist = None  # This is okay, since radar updates can work with "None" intersection points
        return i_dist

    def update_body_collision(self) -> bool:
        """

        :return: boolean if collision with any of the obstacles
        """
        col_bool_list = []
        if len(self.spheres()) > 0:
            col_bool_list.append(
                collision_sphere_spheres(pos1=self.auv.position, rad1=self.auv.safety_radius,
                                         pos2=self.spheres.position, rad2=self.spheres.radius)
            )
        for capsule in self.capsules:
            col_bool_list.append(
                collision_capsule_sphere(cap1=capsule.vec_bot, cap2=capsule.vec_top, cap_rad=capsule.radius,
                                         sph_pos=self.auv.position, sph_rad=self.auv.safety_radius)
            )
        return any(col_bool_list)

    def observe(self) -> np.ndarray:
        obs = np.zeros(self.n_observations, dtype=np.float32)
        # Distance from goal, contained within max_dist_from_goal before done
        obs[0] = np.clip(1 - (np.log(self.delta_d / self.max_dist_from_goal) / np.log(
            self.dist_goal_reached_tol / self.max_dist_from_goal)), 0, 1)
        # Pitch error delta_psi, will be between +90° and -90°
        obs[1] = np.clip(self.delta_theta / (np.pi / 2), -1, 1)
        # Heading error delta_theta, will be between -180 and +180 degree, observation jump is not fixed here,
        # since it might be good to directly indicate which way to turn is faster to adjust heading
        obs[2] = np.clip(self.delta_psi / np.pi, -1, 1)
        obs[3] = np.clip(self.auv.relative_velocity[0] / self.u_max, -1, 1)  # Surge Forward speed
        obs[4] = np.clip(self.auv.relative_velocity[1] / self.v_max, -1, 1)  # Sway Side speed
        obs[5] = np.clip(self.auv.relative_velocity[2] / self.w_max, -1, 1)  # Heave Vertical speed
        obs[6] = np.clip(self.auv.attitude[0] / self.max_attitude, -1, 1)  # Roll
        obs[7] = np.clip(self.auv.attitude[1] / self.max_attitude, -1, 1)  # Pitch
        obs[8] = np.clip(np.sin(self.auv.attitude[2]), -1, 1)  # Yaw, expressed in two polar values to make
        obs[9] = np.clip(np.cos(self.auv.attitude[2]), -1, 1)  # sure observation does not jump between -1 and 1
        obs[10] = np.clip(self.auv.angular_velocity[0] / self.p_max, -1, 1)  # Angular Velocities, roll rate
        obs[11] = np.clip(self.auv.angular_velocity[1] / self.q_max, -1, 1)  # pitch rate
        obs[12] = np.clip(self.auv.angular_velocity[2] / self.r_max, -1, 1)  # Yaw rate
        obs[13] = np.clip(self.nu_c[0] / 2, -1, 1)  # Assuming in general current max. speed of 2m/s
        obs[14] = np.clip(self.nu_c[1] / 2, -1, 1)
        obs[15] = np.clip(self.nu_c[2] / 2, -1, 1)
        obs[self.n_obs_without_radar:] = np.clip(self.radar.intersec_dist / self.radar.max_dist, 0, 1)
        return obs

    def reward_step(self, action: np.ndarray) -> float:
        """
        Calculate the reward function, make sure to call self.is_done() before to update and check the done conditions

        The factors are defined in the config. Each reward is normalized between 0..1, thus the factor decides its
        importance. Keep in mind the rewards for the done conditions will be sparse.

        Reward 1: Navigation Errors
        Reward 2: Stable attitude
        Reward 3: time step penalty
        Reward 4: action use penalty
        Reward 5: Done - Goal reached
        Reward 6: Done - out of bounds position
        Reward 7: Done - out of bounds attitude
        Reward 8: Done - maximum episode steps
        Reward 9: Done - collision

        :param action: array with actions between -1 and 1
        :return: The single reward at this step
        """
        # TODO: Add reward for the collision detection and obstacle closeness
        #  (maybe need complex function that takes the distance to goal into account)
        # TODO: Add continuous reward function for speed, angular and attitude
        # Reward for being closer to the goal location (with old observations):
        self.last_reward_arr[0] = (
                -self.reward_factors["w_d"] * Reward.log_precision(x=self.delta_d,
                                                                   x_goal=self.dist_goal_reached_tol,
                                                                   x_max=self.max_dist_from_goal)
                - self.reward_factors["w_delta_theta"] * (np.abs(self.delta_theta) / (np.pi / 2)) ** 2
                - self.reward_factors["w_delta_psi"] * (np.abs(self.delta_psi) / np.pi) ** 2
        )
        # Reward for stable attitude
        self.last_reward_arr[1] = -self.reward_factors["w_phi"] * (
                np.abs(self.auv.attitude[0]) / self.max_attitude) ** 2 \
                                  - self.reward_factors["w_theta"] * (
                                          np.abs(self.auv.attitude[1]) / self.max_attitude) ** 2
        # Negative cum_reward per time step
        self.last_reward_arr[2] = -self.reward_factors["w_t"]
        # Reward for action used (e.g. want to minimize action power usage), factors can be scalar or matching array
        self.last_reward_arr[3] = - (np.sum(
            np.abs(action) / self.auv.u_bound.shape[0] * self.action_reward_factors)) ** 2

        # Add extra reward on checking which condition caused the episode to be done
        self.last_reward_arr[4:] = np.array(self.conditions) * self.w_done

        # Extra reward based on how the goal has been reached! # TODO: add attitude, make this logarithmic more reward for small speed?
        if self.conditions[0]:
            self.last_reward_arr[4] += (
                    + self.reward_factors["w_goal_pdot"] * Reward.sq_goal_constraints(
                x=np.linalg.norm(self.auv.position_dot),
                l=self.velocity_goal_reached_tol)
                    + self.reward_factors["w_goal_Thetadot"] * Reward.sq_goal_constraints(
                x=np.linalg.norm(self.auv.euler_dot),
                l=self.ang_rate_goal_reached_tol)
            )
        # Just for analyzing purpose:
        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr

        reward = float(np.sum(self.last_reward_arr))

        return reward

    def is_done(self) -> Tuple[bool, list]:
        """
        Condition 0: Check if close to the goal
        Condition 1: Check if out of bounds for position
        Condition 2: Check if attitude (pitch, roll) too high
        Condition 3: Check if maximum time steps reached
        Condition 4: Check for collision

        :return: [if simulation is done, indexes of conditions that are true]
        """
        # All conditions in a list
        self.conditions = [
            # Condition 0: Check if close to the goal
            self.delta_d < self.dist_goal_reached_tol,
            # Condition 1: Check if out of bounds for position
            self.delta_d > self.max_dist_from_goal,
            # Condition 2: Check if attitude (pitch, roll) too high
            np.any(np.abs(self.auv.attitude[:2]) > self.max_attitude),
            # Condition 3: Check if maximum time steps reached
            self.t_steps >= self.max_timesteps,
            # Condition 4: Collision with obstacle (is updated earlier)
            self.collision
        ]

        # If goal reached
        if self.conditions[0]:
            self.goal_constraints = [
                np.linalg.norm(self.auv.position_dot) < self.velocity_goal_reached_tol,
                np.linalg.norm(self.auv.euler_dot) < self.ang_rate_goal_reached_tol,
                (-self.attitude_goal_reached_tol <= geom.ssa(
                    self.auv.attitude[2] - self.heading_goal_reached) <= +self.attitude_goal_reached_tol),
                all((-self.attitude_goal_reached_tol <= self.auv.attitude[:2]) & (
                        self.auv.attitude[:2] <= self.attitude_goal_reached_tol))
            ]
            if all(self.goal_constraints):
                self.goal_reached = True

        # Return also the indexes of which cond is activated
        cond_idx = [i for i, x in enumerate(self.conditions) if x]

        # Check if any condition is true
        done = bool(np.any(self.conditions))  # To satisfy environment checker
        return done, cond_idx

    def render(self, mode="human", rotate_cam=False, real_time=False):
        """

        :param mode: from base class, only human mode in this case
        :param rotate_cam: if rotating the cam slowly (helps with depth in matplotlib)
        :param real_time: if render should approx happen in real time
        :return: None
        """
        if real_time:
            plt.pause(self.t_step_size * 0.9)

        if self.episode_data_storage is None:
            self.init_episode_storage()  # The data storage is needed for the plot
        if self.episode_animation is None:
            self.episode_animation = EpisodeAnimation()
            self.ax = self.episode_animation.init_path_animation()
            self.episode_animation.add_episode_text(self.ax, self.episode)
            # Add goal location as tiny sphere, this one is not an obstacle!
            self.episode_animation.add_shapes(self.ax, [shape.Sphere(self.goal_location, 0.15)], 'r')
            # Add obstacles
            self.episode_animation.add_shapes(self.ax, self.obstacles, 'b')
            # Add radar
            self.episode_animation.init_radar_animation(self.radar.n_rays)

        self.episode_animation.update_path_animation(positions=self.episode_data_storage.positions,
                                                     attitudes=self.episode_data_storage.attitudes)
        self.episode_animation.update_radar_animation(self.radar.pos, self.radar.end_pos_n)
        # Rotate camera:
        if rotate_cam:
            self.episode_animation.ax_path.azim += 1
            # ax.view_init(elev=10., azim=ii)

        # Possible implementation for rgb_array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array,
        # but not really needed here since 3d.

    def save_full_data_storage(self):
        """
        Call this function to save the full data storage
        """
        self.full_data_storage.save()

    def init_episode_storage(self):
        """
        Small helper function for setting up episode storage when needed
        """
        self.episode_data_storage = EpisodeDataStorage()
        self.episode_data_storage.set_up_episode_storage(path_folder=self.save_path_folder, vehicle=self.auv,
                                                         step_size=self.t_step_size, nu_c_init=self.nu_c,
                                                         shapes=[*self.obstacles,
                                                                 shape.Sphere(self.goal_location, 0.15)],
                                                         radar=self.radar, title=self.title,
                                                         episode=self.episode, env=self)

    def generate_random_pos(self, d: float):
        """
        Function to generate random position with distance away from goal

        :param d: Distance spawned away from goal
        :return: array(3,) with random position
        """
        rnd_arr_pos = (np.random.random(3) - 0.5)
        return self.goal_location + rnd_arr_pos * (d / np.linalg.norm(rnd_arr_pos))

    def generate_random_att(self, max_att_factor: float = 0.7):
        rnd_arr_attitude = (np.random.random(3) - 0.5) * 2
        att_factor = np.array([self.max_attitude * max_att_factor,
                               self.max_attitude * max_att_factor,
                               np.pi])  # Spawn at xx% of max attitude
        return rnd_arr_attitude * att_factor  # Spawn with random attitude


class Reward:
    """
    Static class to group the different reward functions
    """

    @staticmethod
    def log_precision(x: float, x_goal: float, x_max: float) -> float:
        """
        Function to scale logarithmic function between x_goal->y=0 and x_max->y=1

        :param x: actual value
        :param x_max: maximum value
        :param x_goal: goal value (smaller than max!)
        :return: x value scaled on log (0 <= x <= 1 when x_goal <= x <= x_max)
        """
        return 1 - (np.log(x / x_max) / np.log(x_goal / x_max))

    @staticmethod
    def disc_goal_constraints(x: float, x_des: float, perc: float = 0.1):
        """
        Function for the final discrete reward when goal is reached for the constraints. This assumes a desired small
        positive value as a target (e.g. tolerance) and the actual achieves value (positive). Overshoot percentage is
        used to make sure, the final result is always below the desired value.

        :param x: actual value
        :param x_des: desired value
        :param perc: percentage to overshoot desired value
        :return: reward [0..2] for actual value from desired value [0..1]
            extra reward when des value actually reached [0,1]
        """
        x_des -= x_des * perc
        return (x_des / max(x_des, x)) ** 2 + (x < x_des)

    @staticmethod
    def cont_goal_constraints(x: float, delta_d: float, x_des: float, delta_d_des: float):
        """
        Continuous reward functions for some goal constraints put into a from goal distance dependent function
        :param x: actual value
        :param delta_d: distance from goal
        :param x_des: desired value
        :param delta_d_des: desired delta distance (needed to form function)
        :return: reward [0..1] for actual value x with delta_d
        """
        return ((1 - (x_des / max(x_des, x))) * (delta_d_des/ max(delta_d_des, delta_d)))**0.5


class SimpleDocking3d(BaseDocking3d):
    """
    This class generates a simple environment to drive in one point in space without obstacles and no current
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios

        """
        DISTANCE_FROM_GOAL = 15

        # Goal location:
        self.goal_location = np.array([0.0, 0.0, 0.0])
        # Goal attitude:
        self.heading_goal_reached = (np.random.random() - 0.5) * np.pi  # Random here, since no capsule here
        # Position
        self.auv.position = self.generate_random_pos(d=DISTANCE_FROM_GOAL)
        # Attitude
        self.auv.attitude = self.generate_random_att(max_att_factor=0.7)
        # Water current
        self.current = Current(mu=0.005, V_min=0.0, V_max=0.0, Vc_init=0.0,
                               alpha_init=0.0, beta_init=0.0, white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)
        # Obstacles:
        self.obstacles = []


class SimpleCurrentDocking3d(SimpleDocking3d):
    """
    This class generates a simple environment to drive in one point in space without obstacles but with custom
    defined current
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """
        # Same setup as before:
        super().generate_environment()
        # Water current
        curr_angle = (np.random.random(2) - 0.5) * 2 * np.array([np.pi / 2, np.pi])  # Water current direction
        self.current = Current(mu=0.005, V_min=0.5, V_max=0.5, Vc_init=0.5,
                               alpha_init=curr_angle[0], beta_init=curr_angle[1], white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)


class CapsuleDocking3d(SimpleDocking3d):
    """
    This class generates an environment only with the capsule to dock at
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """
        CAPSULE_RADIUS = 1.0
        CAPSULE_HEIGHT = 4.0

        # Same setup as before:
        super().generate_environment()
        # Goal location: Random around the shaft of the capsule + auv radius + margin to make it even possible to
        # reach the goal before colliding with the capsule
        theta = np.random.rand() * 2 * np.pi
        radius = CAPSULE_RADIUS + self.auv.safety_radius
        x, y = np.cos(theta) * radius, np.sin(theta) * radius
        self.goal_location = np.array([x,
                                       y,
                                       (np.random.rand() - 0.5) * CAPSULE_HEIGHT])
        # Obstacles (only the capsule at goal location):
        cap = Capsule(position=np.array([0.0, 0.0, 0.0]),
                      radius=CAPSULE_RADIUS,
                      vec_top=np.array([0.0, 0.0, -CAPSULE_HEIGHT / 2.0]))
        self.capsules = [cap]
        self.obstacles = [*self.capsules]
        # Get vector pointing from goal location to capsule:
        vec = shape.vec_line_point(self.goal_location, cap.vec_top, cap.vec_bot)
        # Calculate heading at goal
        self.heading_goal_reached = geom.ssa(np.arctan2(vec[1], vec[0]))


class CapsuleCurrentDocking3d(CapsuleDocking3d):
    """
    This class generates an environment only with the capsule to dock at and with current
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """
        # Same setup as before:
        super().generate_environment()
        # Water current
        curr_angle = (np.random.random(2) - 0.5) * 2 * np.array([np.pi / 2, np.pi])  # Water current direction
        self.current = Current(mu=0.005, V_min=0.5, V_max=0.5, Vc_init=0.5,
                               alpha_init=curr_angle[0], beta_init=curr_angle[1], white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)


class ObstaclesDocking3d(CapsuleDocking3d):
    """
    This class generates an environment already with multiple obstacles
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up and environment with multiple capsules as obstacles around the goal location (e.g. a pear or oil rig)
        """
        # TODO: ADD strategy for spawn location to be away from capsules! (maybe to z location limitation, do not forget to update reset function then)
        CAPSULE_OBSTACLES_RADIUS = 1.0
        CAPSULE_OBSTACLES_HEIGHT = 2 * self.max_dist_from_goal
        CAPSULE_DISTANCE_FROM_CENTER = 6
        NUMBER_OF_CAPSULES = 4

        # Same setup as before:
        super().generate_environment()

        # Generate new capsules
        new_capsules = []
        theta = np.random.rand() * 2 * np.pi
        for i in range(NUMBER_OF_CAPSULES):
            x = np.cos(theta) * CAPSULE_DISTANCE_FROM_CENTER
            y = np.sin(theta) * CAPSULE_DISTANCE_FROM_CENTER
            theta += 2 * np.pi / NUMBER_OF_CAPSULES
            # Create new capsule and append to list
            new_capsules.append(
                Capsule(position=np.array([x, y, 0.0]),
                        radius=CAPSULE_OBSTACLES_RADIUS,
                        vec_top=np.array([x, y, -CAPSULE_OBSTACLES_HEIGHT / 2.0]))
            )
        # Add Obstacles:
        self.capsules.extend(new_capsules)
        self.obstacles.extend(new_capsules)


class ObstaclesCurrentDocking3d(ObstaclesDocking3d):
    """
    Set up and environment with multiple capsules as obstacles around the goal location (e.g. a pear or oil rig)
    Add water current to it
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)

    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """
        # Same setup as before:
        super().generate_environment()
        # Water current
        curr_angle = (np.random.random(2) - 0.5) * 2 * np.array([np.pi / 2, np.pi])  # Water current direction
        self.current = Current(mu=0.005, V_min=0.5, V_max=0.5, Vc_init=0.5,
                               alpha_init=curr_angle[0], beta_init=curr_angle[1], white_noise_std=0.0,
                               step_size=self.auv.step_size)
        self.nu_c = self.current(self.auv.attitude)
