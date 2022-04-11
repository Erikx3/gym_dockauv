import os
import pickle
from copy import deepcopy
import datetime
import numpy as np

# Used for typehints
from ..objects.auvsim import AUVSim
from ..objects.shape import Shape
from typing import List


class FullDataStorage:
    """
    TODO:
    Class to save general simulation over all the runs of simulation (e.g. length, success/collison, reward)
    """
    pass


class ArrayList:
    """
    Custom data type that works as fast as a python list with memory optimization as in numpy

    Background: np.append, np.concatenate, np.vstack are very slow, however, it is necessary for the storage classes
    to grow their array size dynamically and represent the data array so far retrieved for the live animation.
    Already with only 100k function calls it takes a minutes for numpy array to concatenate, while appending list
    like or preallocation takes 0.5 seconds here. More on that here:
    https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    https://stackoverflow.com/questions/46102225/which-one-is-faster-np-vstack-np-append-np-concatenate-or-a-manual-function-ma

    Arrays are saved based on dimension of first vector in nxc format, where c i the length of the initial vector

    :param init_array: 1d array with length c
    """

    def __init__(self, init_array):
        # Initialize data array
        self.dim_col = len(init_array)
        self.capacity = 100
        self.data = np.zeros((self.capacity, self.dim_col))
        self.size = 1
        self.data[0, :] = init_array

        # Some settings:
        self.array_grow_factor = 4

    def add_row(self, row):
        if self.size == self.capacity:
            self.capacity *= self.array_grow_factor
            newdata = np.zeros((self.capacity, self.dim_col))
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = row
        self.size += 1

    def get_data(self):
        return self.data[:self.size]


class EpisodeDataStorage:
    r"""
    Class to save data e.g. vehicle related data during a simulation (e.g. one episode). Initialize and update after
    all other initialization and updates

    This data is saved in dictionaries, the keys for retrieval are going to be defined here.

    For the definition of the arrays, please refer to the vehicle documentation.

    Pickle Structure (will be one dict):
    {
        "vehicle":
        {
            "object": auvsim.instance (initial one),
            "states": auvsim.state nx12 array,
            "states_dot": auvsim._state_dot nx12 array,
            "u": auvsim.u nxa array (a number of action),
            "shapes": shape object as in shapes.py used here
        },
        ... TODO (Agent, environment, further variables, settings etc. or these go to FullDataStorge class)

    :param path_folder: Path to folder
    :param vehicle: Vehicle that is simulated
    :param shapes: Shapes for 3d Animation that were used (static)
    :param title: Optional title for addendum
    :param episode: Episode number

    filename: will be formatted path_folder\YYYY-MM-DDTHH-MM-SS__episode{episode}__{title}.pkl
    }
    """

    def __init__(self, path_folder: str, vehicle: AUVSim, shapes: List[Shape] = None,
                 title: str = "", episode: int = -1):
        # Some variables for saving the file
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_M_%S')
        self.filename = os.path.join(path_folder, f"{utc_str}__episode{episode}__{title}.pkl")
        self.vehicle = vehicle  # Vehicle instance (not a copy, automatically a reference which is updated in reference)
        if shapes is None:
            shapes = []
        self.storage = {
            "vehicle": {
                "object": vehicle,
                "states": ArrayList(self.vehicle.state),
                "states_dot": ArrayList(self.vehicle._state_dot),
                "u": ArrayList(self.vehicle.u),
            },
            "shapes": [deepcopy(shape) for shape in shapes],
            "episode": episode
        }

    def update(self) -> None:
        """
        should be called in the end of each simulation step

        :return: None
        """
        self.storage["vehicle"]["states"].add_row(self.vehicle.state)
        self.storage["vehicle"]["states_dot"].add_row(self.vehicle._state_dot)
        self.storage["vehicle"]["u"].add_row(self.vehicle.u)

    def save(self) -> None:
        """
        Function to save the pickle file
        """
        self.storage["vehicle"]["states"] = self.storage["vehicle"]["states"].get_data()
        self.storage["vehicle"]["states_dot"] = self.storage["vehicle"]["states_dot"].get_data()
        self.storage["vehicle"]["u"] = self.storage["vehicle"]["u"].get_data()

        with open(self.filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.storage, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str):
        with open(file_name, 'rb') as inp:
            load_file = pickle.load(inp)
            return load_file

    @property
    def positions(self):
        r"""
        Returns all the position of the AUV in NED coordinates x, y, z

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"].get_data()[:, 0:3]

    @property
    def attitudes(self):
        r"""
        Returns all the attitude (euler angles) of the AUV wrt. to NED coordinates roll, pitch, yaw.

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"].get_data()[:, 3:6]
