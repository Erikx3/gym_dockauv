"""
This file serves as a template and default config, when e.g. gym.make is called
"""
import numpy as np
import os
import copy

# --------- Registration dictionary for gym environments ----------
REGISTRATION_DICT = {
    "SimpleDocking3d-v0": "gym_dockauv.envs:SimpleDocking3d",
    "SimpleCurrentDocking3d-v0": "gym_dockauv.envs:SimpleCurrentDocking3d"
}

# ---------- Base Config for all other possible configs ----------
BASE_CONFIG = {
    # ---------- GENERAL ----------
    "config_name": "DEFAULT_BASE_CONFIG",   # Optional Identifier of config, if the title does not say everything
    "title": "DEFAULT",                     # Title used in e.g. animation plot or file names as identifier
    "log_level": 20,                        # Level of logging, 0 means all messages, 30 means warning and above,
                                            # look here: https://docs.python.org/3/library/logging.html#logging-levels
    "verbose": 1,                           # If logs should also appear in output console, either 0 or 1

    # ---------- EPISODE ----------
    "max_timesteps": 500,                   # Maximum amount of timesteps before episode ends

    # ---------- SIMULATION --------------
    "t_step_size": 0.10,                    # Length of each simulation timestep [s]
    "interval_datastorage": 200,            # Interval of episodes on which extended data is saved through data class
    "save_path_folder": os.path.join(os.getcwd(), "logs"),  # Folder name where all result files will be stored

    # ---------- GOAL AND DONE----------
    "goal_location": np.array([0, 0, 0]),   # TODO: Think about moving goal location to gym generate_env
    "max_dist_from_goal": 10,               # Maximum distance away from goal before simulation end
    "max_attitude": 60/180*np.pi,           # Maximum attitude allowed for vehicle

    # ---------- AUV & REWARDS ----------
    "vehicle": "BlueROV2",                  # Name of the vehicle, look for available vehicles in
                                            # gym_dockauv/objects/vehicles
    "radius": 0.5,                          # Radius size of vehicle for collision detection
    "reward_factors":
        np.array([-0.7, -0.6, -0.05, -0.4,  # Reward factors for each reward, look into the reward step doc for more
                  50, -100, -100, -50]),    # details
    "action_reward_factors": np.array(
        [0.05, 0.15, 0.05,                  # Negative reward factors for each possible action input used, must be
         0.25, 0.25, 0.25]),                 # changed depending on actionspace of vehicle

    # --------- RADAR -----------  Will be dynamically loaded via **kwargs
    "radar": {
        "freq": 1,                         # Frequency of updates of radars TODO: Not yet implemented
        "alpha": 20 * np.pi / 180,          # Range of vertical angle wideness
        "beta": 20 * np.pi / 180,           # Range of horizontal angle wideness
        "ray_per_deg": 5 * np.pi / 180,     # rad inbetween each ray, must leave zero remainder with alpha and beta
        "max_dist": 2                       # Maximum distance the radar can look ahead
    }
}

# ---------- Configuration for Training runs ----------
TRAIN_CONFIG = copy.deepcopy(BASE_CONFIG)
TRAIN_CONFIG["title"] = "Training Run"
TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs")

# ---------- Configuration for Prediction runs ----------
PREDICT_CONFIG = copy.deepcopy(BASE_CONFIG)
PREDICT_CONFIG["interval_datastorage"] = 1
PREDICT_CONFIG["title"] = "Prediction Run"
PREDICT_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "predict_logs")
# PREDICT_CONFIG["max_dist_from_goal"] = 10

# ---------- Configuration for Manual control runs ----------
MANUAL_CONFIG = copy.deepcopy(BASE_CONFIG)
MANUAL_CONFIG["title"] = "Manual Run"
MANUAL_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "manual_logs")
MANUAL_CONFIG["interval_datastorage"] = 1
#MANUAL_CONFIG["max_timesteps"] = 200000

