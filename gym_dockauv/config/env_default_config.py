"""
This file serves as a template and default config, when e.g. gym.make is called
"""
import numpy as np

BASE_CONFIG = {
    # ---------- EPISODE ----------
    "max_timesteps": 500,  # Maximum amount of timesteps before episode ends

    # ---------- SIMULATION --------------
    "t_step_size": 0.2,  # Length of each simulation timestep [s]
    "interval_datastorage": 100,  # Interval of episodes on which extended data is saved through data class
    "interval_render": 20,  # Interval of episodes on which the simulation should be rendered

    # ---------- GOAL ----------
    "goal_location": np.array([0, 0, 0]),

    # ---------- AUV ----------
    'vehicle': "BlueROV2",  # Name of the vehicle, look for available vehicles in gym_dockauv/objects/vehicles
    'radius': 0.5  # Radius size of vehicle for collision detection
}
