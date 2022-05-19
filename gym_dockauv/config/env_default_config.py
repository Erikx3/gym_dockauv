"""
This file serves as a template and default config, when e.g. gym.make is called
"""
import numpy as np
import os
import copy

BASE_CONFIG = {
    # ---------- GENERAL ----------
    "config_name": "DEFAULT_BASE_CONFIG",   # Optional Identifier of config
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
    "max_dist_from_goal": 10,               # Maximum distance away from goal before simulation end

    # ---------- GOAL ----------
    "goal_location": np.array([0, 0, 0]),

    # ---------- AUV ----------
    'vehicle': "BlueROV2",                  # Name of the vehicle, look for available vehicles in
                                            # gym_dockauv/objects/vehicles
    'radius': 0.5                           # Radius size of vehicle for collision detection
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

# ---------- Configuration for Manual control runs ----------
MANUAL_CONFIG = copy.deepcopy(BASE_CONFIG)
MANUAL_CONFIG["title"] = "Manual Run"
MANUAL_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "manual_logs")
