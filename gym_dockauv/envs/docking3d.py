import datetime
import importlib
import logging
import os
import pprint
import time
from timeit import default_timer as timer
from typing import Tuple, Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from gym_dockauv.config.env_default_config import BASE_CONFIG
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation

# TODO: Think about making this a base class for further environments with different observations, rewards, setup?
# TODO: Save cumulative reward in episode data storage, other information in Simulation Storage
# TODO: Save animation option
# TODO: Water current, radar sensors, obstacles (so far only capsules are supported)
# TODO: Log rewards and other also in pkl file, make it analyzeable

# Set logger
logger = logging.getLogger(__name__)


class Docking3d(gym.Env):
    """
    Base Class for the docking environment
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__()
        # Basic config for logger
        self.config = env_config
        self.title = self.config["title"]
        self.save_path_folder = self.config["save_path_folder"]
        self.log_level = self.config["log_level"]
        self.verbose = self.config["verbose"]
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
        self.auv = AUV()

        # Set step size for vehicle
        self.auv.step_size = self.config["t_step_size"]

        # Set the action and observation space
        self.action_space = gym.spaces.Box(low=self.auv.u_bound[:, 0],
                                           high=self.auv.u_bound[:, 1],
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.ones(12),
                                                high=np.ones(12),
                                                dtype=np.float32)

        # General simulation variables:
        self.t_total_steps = 0  # Number of steps in this simulation
        self.t_step_size = self.config["t_step_size"]
        self.episode = 0  # Current episode
        self.cumulative_reward = 0  # Current cumulative reward of agent
        self.max_timesteps = self.config["max_timesteps"]
        self.interval_datastorage = self.config["interval_datastorage"]

        # Goal
        self.goal_location = self.config["goal_location"]

        # Declaring attributes
        self.obstacles = []
        self.goal_reached = False  # Bool to check of goal is reached at the end of an episode
        self.collision = False  # Bool to indicate of vehicle has collided

        # Initialize observation, reward, done, info
        self.n_observations = 12
        self.observation = np.zeros(self.n_observations)
        self.done = False
        self.last_reward = 0
        self.last_reward_arr = np.zeros(7)  # This should reflect the dimension of the rewards parts (for analysis)
        self.cum_reward_arr = np.zeros(7)
        self.info = {}
        self.conditions = None  # Boolean array to see which conditions are true

        # Water current TODO
        self.nu_c = np.zeros(6)

        # Save and display simulation time
        self.start_time_sim = timer()

        # Episode Data storage
        self.episode_data_storage = None

        # Full data storage:
        self.full_data_storage = FullDataStorage()
        self.full_data_storage.set_up_episode_storage(env=self, path_folder=self.save_path_folder, title=self.title)

        # Animation
        self.episode_animation = None

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

        .. note:: Options not used yet
        """
        # In case any windows were open from matplotlib or animation
        if self.episode_animation:
            plt.close(self.episode_animation.fig)  # TODO: Window stays open, prob due to Gym
            self.episode_animation = None

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
        self.t_total_steps = 0
        self.goal_reached = False
        self.collision = False

        # Reset observation, cum_reward, done, info
        self.observation = self.auv.state
        self.last_reward = 0
        self.cumulative_reward = 0
        self.done = False
        self.last_reward_arr = np.zeros(7)
        self.cum_reward_arr = np.zeros(7)
        self.info = {}
        self.conditions = None  # Boolean array to see which conditions are true

        # Update the seed:
        # TODO: Check if this makes all seeds same (e.g. for water current!!) or works in general
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Update episode number
        self.episode += 1

        # ----------- Init for new environment from here on -----------
        # Generate environment
        self.generate_environment()

        # Save whole episode data if interval is met, or we need it for the renderer
        if self.episode % self.interval_datastorage == 0 or self.episode == 1:
            self.init_episode_storage()
        else:
            self.episode_data_storage = None

        # Log the episode
        logger.info("Environment reset call: \n" + pprint.pformat(return_info_dict))

        # Return info if wanted
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        """
        Setup a environment after each reset
        """
        # TODO Think about how this should be done in future simulations
        rnd_arr = (np.random.random(3) - 0.5)
        self.auv.position = rnd_arr * (8 / np.linalg.norm(rnd_arr))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Simulate current TODO

        # Update AUV dynamics
        self.auv.step(action, self.nu_c)

        # Check collision TODO

        # Update data storage if active
        if self.episode_data_storage:
            self.episode_data_storage.update(self.nu_c, self.cum_reward_arr, self.last_reward_arr)

        # Update visualization if active
        if self.episode_animation:
            self.render()

        # Determine if simulation is done, this also updates self.last_reward
        self.done, cond_idx = self.is_done()

        # Calculate rewards
        self.last_reward = self.reward_step()
        self.cumulative_reward += self.last_reward

        # Make next observation TODO
        self.observation = self.observe()

        # Save sim time info
        self.t_total_steps += 1

        # Update info dict
        self.info = {"episode_number": self.episode,  # Need to be episode number, because episode is used by sb3
                     "t_step": self.t_total_steps,
                     "cumulative_reward": self.cumulative_reward,
                     "last_reward": self.last_reward,
                     "done": self.done,
                     "conditions_true": cond_idx,
                     "collision": self.collision,
                     "goal_reached": self.goal_reached,
                     "simulation_time": timer() - self.start_time_sim}

        return self.observation, self.last_reward, self.done, self.info

    def observe(self) -> np.ndarray:
        diff = self.goal_location - self.auv.position
        obs = np.zeros(self.n_observations)
        obs[0] = np.clip(diff[0] / 50, -1, 1)  # Position difference, assuming we move withing 50 meters
        obs[1] = np.clip(diff[1] / 50, -1, 1)
        obs[2] = np.clip(diff[2] / 50, -1, 1)
        obs[3] = np.clip(self.auv.relative_velocity[0] / 5, -1, 1)  # Forward speed, assuming 5m/s max
        obs[4] = np.clip(self.auv.relative_velocity[1] / 2, -1, 1)  # Side speed, assuming 5m/s max
        obs[5] = np.clip(self.auv.relative_velocity[2] / 2, -1, 1)  # Vertical speed, assuming 5m/s max
        obs[6] = np.clip(self.auv.attitude[0] / np.pi / 2, -1, 1)  # Roll, assuming +-90deg max
        obs[7] = np.clip(self.auv.attitude[1] / np.pi / 2, -1, 1)  # Pitch, assuming +-90deg max
        obs[8] = np.clip(self.auv.attitude[2] / np.pi, -1, 1)  # Yaw, assuming +-180deg max
        obs[9] = np.clip(self.auv.angular_velocity[0] / 1.0, -1, 1)  # Angular Velocities, assuming 1 rad/s
        obs[10] = np.clip(self.auv.angular_velocity[1] / 1.0, -1, 1)
        obs[11] = np.clip(self.auv.angular_velocity[2] / 1.0, -1, 1)

        return obs

    def reward_step(self) -> float:
        """
        Calculate the reward function, make sure to call self.is_done() before to update and check the done conditions

        Reward 1: Close gto goal location
        Reward 2: Stable attitude
        Reward 3: time step penalty
        Reward 4: Done - Goal reached
        Reward 5: Done - out of bounds position
        Reward 6: Done - out of bounds attitude
        Reward 7: Done - maximum episode steps

        :return: The single reward at this step
        """
        # Reward for being closer to the goal location:
        self.last_reward_arr[0] = -np.linalg.norm(self.auv.position - self.goal_location) ** 2.0
        # Reward for stable attitude
        self.last_reward_arr[1] = -np.sum(np.abs(self.auv.attitude[:2])) * 5
        # Negative cum_reward per time step
        self.last_reward_arr[2] = -5

        # Reward for action used (e.g. want to minimize action power usage) TODO

        # Add extra reward on checking which condition caused the episode to be done
        self.last_reward_arr[3:] = np.array([50000, 0, 0, 0]) * np.array(self.conditions)

        # Just for analyzing purpose:
        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr

        reward = float(np.sum(self.last_reward_arr))

        return reward

    def is_done(self) -> Tuple[bool, list]:
        """
        Condition 0: Check if close to the goal
        Condition 1: Check if out of bounds for position
        Condition 2: Check if attitude (pitch, roll) too high
        Condition 3: # Check if maximum time steps reached

        :return: [if simulation is done, extra discrete reward, indexes of conditions that are true]
        """
        # TODO: Collision

        # All conditions in a list
        self.conditions = [
            np.linalg.norm(self.auv.position - self.goal_location) < 1.0,  # Condition 0: Check if close to the goal
            np.any(np.abs(self.auv.position) > 50),  # Condition 1: Check if out of bounds for position
            np.any(np.abs(self.auv.attitude[:2]) > 80 / 180 * np.pi),
            # Condition 2: Check if attitude (pitch, roll) too high
            self.t_total_steps >= self.max_timesteps  # Condition 3: # Check if maximum time steps reached
        ]

        # If goal reached
        if self.conditions[0]:
            self.goal_reached = True

        # Return also the indexes of which cond is activated
        cond_idx = [i for i, x in enumerate(self.conditions) if x]

        # Check if any condition is true
        done = np.any(self.conditions)
        return done, cond_idx

    def render(self, mode="human", real_time=False):
        if self.episode_data_storage is None:
            self.init_episode_storage()  # The data storage is needed for the plot
        if self.episode_animation is None:
            self.episode_animation = EpisodeAnimation()
            ax = self.episode_animation.init_path_animation()
            self.episode_animation.add_episode_text(ax, self.episode)

        self.episode_animation.update_path_animation(positions=self.episode_data_storage.positions,
                                                     attitudes=self.episode_data_storage.attitudes)
        if real_time:
            plt.pause(self.t_step_size * 0.9)

        # Possible implementation for rgb_array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array,
        # but not really needed here since 3d.

    def save(self):
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
                                                         shapes=self.obstacles, radar=None, title=self.title,
                                                         episode=self.episode, cum_rewards=self.cum_reward_arr,
                                                         rewards=self.last_reward_arr)
