"""
This file serves as a template and default config, when e.g. gym.make is called
"""
import numpy as np
import os
import copy

# --------- Registration dictionary for gym environments ----------
REGISTRATION_DICT = {
    "SimpleDocking3d-v0": "gym_dockauv.envs:SimpleDocking3d",
    "SimpleCurrentDocking3d-v0": "gym_dockauv.envs:SimpleCurrentDocking3d",
    "CapsuleDocking3d-v0": "gym_dockauv.envs:CapsuleDocking3d",
    "CapsuleCurrentDocking3d-v0": "gym_dockauv.envs:CapsuleCurrentDocking3d",
    "ObstaclesDocking3d-v0": "gym_dockauv.envs:ObstaclesDocking3d",
    "ObstaclesCurrentDocking3d-v0": "gym_dockauv.envs:ObstaclesCurrentDocking3d"
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
    "interval_episode_log": 50,             # Log the episode info dict in specific interval into the log file
    "save_path_folder": os.path.join(os.getcwd(), "logs"),  # Folder name where all result files will be stored

    # ---------- GOAL AND DONE----------
    "max_dist_from_goal": 20,               # Maximum distance away from goal before simulation end
    "max_attitude": 60/180*np.pi,           # Maximum attitude allowed for vehicle
    "dist_goal_reached": 0.1,               # Distance within it counts as goal successfully reached

    # ---------- AUV & REWARDS ----------
    "vehicle": "BlueROV2",                  # Name of the vehicle, look for available vehicles in
                                            # gym_dockauv/objects/vehicles
                                            # Observation normalization parameters
    "u_max": 2.0,                           # Surge max
    "v_max": 1.5,                           # Sway max
    "w_max": 1.5,                           # Heave max
    "p_max": 90 * np.pi/180,                # Roll rate max
    "q_max": 90 * np.pi/180,                # Pitch rate max
    "r_max": 120 * np.pi/180,               # Yaw rate max
    "radius": 0.5,                          # Radius size of vehicle for collision detection
    "reward_factors":
        np.array(
            [-0.7, -0.6, -0.05, -0.4,       # Reward factors for each reward, look into the reward step doc for more
             50, -100, -100, -50, -100]),   # details
    "action_reward_factors": 1,             # Negative reward factor for action, can be an array matching the number of
                                            # actions or just a scalar multiplied with the normalized sum of the action

    # --------- RADAR -----------  Will be dynamically loaded via **kwargs
    "radar": {
        "freq": 1,                         # Frequency of updates of radars TODO: Not yet implemented
        "alpha": 70 * np.pi / 180,          # Range of vertical angle wideness
        "beta": 70 * np.pi / 180,           # Range of horizontal angle wideness
        "ray_per_deg": 10 * np.pi / 180,     # rad inbetween each ray, must leave zero remainder with alpha and beta
        "max_dist": 5                       # Maximum distance the radar can look ahead
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
PREDICT_CONFIG["interval_episode_log"] = 1
# PREDICT_CONFIG["max_dist_from_goal"] = 10

# ---------- Configuration for Manual control runs ----------
MANUAL_CONFIG = copy.deepcopy(BASE_CONFIG)
MANUAL_CONFIG["title"] = "Manual Run"
MANUAL_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "manual_logs")
MANUAL_CONFIG["interval_datastorage"] = 1
MANUAL_CONFIG["interval_episode_log"] = 1
#MANUAL_CONFIG["max_timesteps"] = 200000

