import numpy as np
import gym
from gym.utils import seeding
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta

import importlib
from typing import Tuple, Optional, Union

from gym_dockauv.utils.datastorage import EpisodeDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.config.env_default_config import BASE_CONFIG

# TODO: Think about making this a base class for further environments with different observations, rewards, setup?
# TODO: Save cumulative reward in episode data storage, other information in Simulation Storage
# TODO: Make (simple) logger with env config, BlueROV2 config, when something is saved etc.
# TODO: Save animation option
# TODO: Water current, radar sensors, obstacles (so far only capsules are supported)


class Docking3d(gym.Env):
    """
    Base Class for the docking environment
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__()

        self.config = env_config

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
        self.interval_render = self.config["interval_render"]
        self.save_path_folder = self.config["save_path_folder"]
        self.real_time = self.config["real_time"]

        # Goal
        self.goal_location = self.config["goal_location"]

        # Declaring attributes
        self.obstacles = []
        self.reached_goal = False  # Bool to check of goal is reached at the end of an episode
        self.collision = False  # Bool to indicate of vehicle has collided

        # Initialize observation, reward, done, info
        self.n_observations = 12
        self.observation = np.zeros(self.n_observations)
        self.done = False
        self.last_reward = 0
        self.info = {}

        # Water current TODO
        self.nu_c = np.zeros(6)

        # Save and display simulation time
        self.start_time_sim = timer()

        # Data storage
        self.episode_data_storage = None

        # Animation
        self.episode_animation = None

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
        plt.close('all')
        self.episode_animation = None

        # Save info to return in the end
        return_info_dict = self.info

        # Check if we should save a datastorage item
        if self.episode_data_storage and (self.episode % self.interval_datastorage == 0 or self.episode == 1):
            self.episode_data_storage.save()
        self.episode_data_storage = None

        # General reset
        self.auv.reset()
        self.t_total_steps = 0
        self.reached_goal = False
        self.collision = False

        # Reset observation, cum_reward, done, info
        self.observation = self.auv.state
        self.last_reward = 0
        self.cumulative_reward = 0
        self.done = False

        # Update the seed:
        # TODO: Check if this makes all seeds same (e.g. for water current!!)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Update episode number
        self.episode += 1

        # Generate environment
        self.generate_environment()

        # Save whole episode data if interval is met, or we need it for the renderer
        if self.episode % self.interval_datastorage == 0 or self.episode == 1 or self.episode % self.interval_render == 0:
            self.episode_data_storage = EpisodeDataStorage()
            self.episode_data_storage.set_up_episode_storage(path_folder=self.save_path_folder, vehicle=self.auv,
                                                             step_size=self.t_step_size, nu_c_init=self.nu_c,
                                                             shapes=self.obstacles, radar=None, title="",
                                                             episode=self.episode)
        else:
            self.episode_data_storage = None

        # Initialize plot if renderer is active:
        if self.episode % self.interval_render == 0:
            self.episode_animation = EpisodeAnimation()
            ax = self.episode_animation.init_path_animation()
            self.episode_animation.add_episode_text(ax, self.episode)

        # Return info if wanted
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        """
        Setup a environment after each reset
        """
        # TODO Think about how this should be done in future simulations
        self.auv.position = np.array([5, 5, 5])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Simulate current TODO

        # Update AUV dynamics
        self.auv.step(action, self.nu_c)

        # Check collision TODO

        # Update data storage if active
        if self.episode_data_storage:
            self.episode_data_storage.update(self.nu_c)

        # Update visualization if active
        if self.episode_animation:
            self.render()

        # Calculate cum_reward
        self.last_reward = self.reward_step()
        self.cumulative_reward += self.last_reward

        # Determine if simulation is done
        self.done = self.is_done()

        # Make next observation TODO
        self.observation = self.observe()

        # Save sim time info
        self.t_total_steps += 1

        # Update info dict
        self.info = {"episode": self.episode,
                     "t_step": self.t_total_steps,
                     "cumulative_reward": self.cumulative_reward,
                     "last_reward": self.last_reward,
                     "done": self.done,
                     "collision": self.collision,
                     "reached_goal": self.reached_goal,
                     "simulation_time": timer()-self.start_time_sim}

        return self.observation, self.last_reward, self.done, self.info

    def observe(self) -> np.ndarray:
        obs = np.zeros(self.n_observations)
        obs[0] = np.clip(self.auv.position[0] / 50, -1, 1)  # Position, assuming we move withing 50 meters
        obs[1] = np.clip(self.auv.position[1] / 50, -1, 1)
        obs[2] = np.clip(self.auv.position[2] / 50, -1, 1)
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
        # Reward for being closer to the goal location:
        reward_dis = -np.linalg.norm(self.auv.position - self.goal_location)
        # Reward for stable attitude
        reward_att = -np.sum(self.auv.attitude[:2])
        # Negative cum_reward per time step
        reward_time = -1
        # Reward if collision TODO

        # Reward for action used (e.g. want to minimize action power usage) TODO

        reward = reward_dis + reward_att + reward_time

        return reward

    def is_done(self) -> bool:
        # Check if close to the goal
        cond1 = np.linalg.norm(self.auv.position - self.goal_location) < 0.1

        # Check if out of bounds for position
        cond2 = np.any(np.abs(self.auv.position) > 50)

        # Check if attitude (pitch, roll) too high
        cond3 = np.any(np.abs(self.auv.attitude[:2]) > 85 / 180 * np.pi)

        # Check if maximum time steps reached
        cond4 = self.t_total_steps >= self.max_timesteps

        # TODO: Collision

        done = cond1 or cond2 or cond3 or cond4
        return done

    def render(self, mode="human"):
        self.episode_animation.update_path_animation(positions=self.episode_data_storage.positions,
                                                     attitudes=self.episode_data_storage.attitudes)
        if self.real_time:
            plt.pause(self.t_step_size * 0.9)

        # Possible implementation for rgb_array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array,
        # but not really needed here since 3d.
