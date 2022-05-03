import numpy as np
import gym
import importlib
from typing import Tuple

from gym_dockauv.objects.vehicles.BlueROV2 import BlueROV2


class Docking3d(gym.Env):

    def __init__(self, env_config: dict):
        super().__init__()

        self.config = env_config

        # Dynamically load class of vehicle and instantiate it (available vehicles under gym_dockauv/objects/vehicles)
        AUV = getattr(importlib.import_module("gym_dockauv.objects.vehicles." + self.config["vehicle"]),
                      self.config["vehicle"])
        # TODO: Comment out again
        self.auv = BlueROV2()
        # self.auv = AUV()

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

        # Declaring attributes
        self.obstacles = []

        self.reached_goal = None  # Bool to check of goal is reached at the end of an episode
        self.collision = None  # Bool to indicate of vehicle has collided

        # Initialize observation, reward, done, info
        self.observation = None
        self.reward = None
        self.done = False
        self.info = None  # TODO: Make this a dictionary I guess :)

    def reset(self) -> np.ndarray:
        """
        Call this function to reset the environment
        """
        self.auv.reset()
        self.t_total_steps = 0
        self.t_step_size = self.config["t_step_size"]
        self.episode = 0
        self.cumulative_reward = 0
        self.reached_goal = False
        self.collision = False

        # Initialize observation, reward, done, info
        self.observation = self.auv.state
        self.reward = 0
        self.done = False
        self.info = None

        return self.observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    def observe(self):
        pass

    def reward(self):
        pass
