"""
This file serves as a template and default config, when e.g. gym.make is called
"""

BASE_CONFIG = {
    # ---------- EPISODE ----------
    "max_timesteps": 500,  # Maximum amount of timesteps before episode ends

    # ---------- SIMULATION --------------
    "t_step_size": 0.2,  # Length of each simulation timestep [s]

    # ---------- AUV ----------
    'vehicle': "BlueROV2",  # Name of the vehicle, look for available vehicles in gym_dockauv/objects/vehicles
    'radius': 0.5  # Radius size of vehicle for collision detection
}
