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

    Arrays are saved based on dimension of first vector in nxc format, where c is the length of the initial vector

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

    def __getitem__(self, index):
        return self.data[:self.size][index]

    def add_row(self, row: np.ndarray) -> None:
        if self.size == self.capacity:
            self.capacity *= self.array_grow_factor
            newdata = np.zeros((self.capacity, self.dim_col))
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = row
        self.size += 1

    def get_nparray(self) -> np.ndarray:
        return self.data[:self.size]


class EpisodeDataStorage:
    r"""
    Class to save data e.g. vehicle related data during a simulation (e.g. one episode). Initialize and update after
    all other initialization and updates

    This data is saved in dictionaries, the keys for retrieval are going to be defined here.

    For the definition of the arrays, please refer to the vehicle documentation.

    .. note:: This structure can also be used to load the data and make it more convenient to retrieve specific data
        again by the property functions. However one must keep in mind: during the simulation process, the data type of
        the arrays in the self.storage dictionary are custom defined arrays and do not offer all possibilities like the
        array

    Pickle Structure (will be one dict):
    {
        "vehicle":
        {
            "object": auvsim.instance (initial one),
            "states": auvsim.state nx12 array,
            "states_dot": auvsim._state_dot nx12 array,
            "u": auvsim.u nxa array (a number of action)
        },
        "shapes": list of shape objects as in shapes.py used here
        "episode": episode number
        "step_size": step_size in simulation
        "nu_c": water current
        ... TODO (Agent, environment, further variables, settings etc. or these go to FullDataStorge class)

    }
    """

    def __init__(self):
        self.storage = None
        self.vehicle = None
        self.file_save_name = None

    def set_up_episode_storage(self, path_folder: str, vehicle: AUVSim, step_size: float, nu_c_init: np.ndarray, shapes: List[Shape] = None,
                               title: str = "", episode: int = -1) -> None:
        r"""
        Set up the storage to save and update incoming data

        file_save_name: will be formatted path_folder\YYYY-MM-DDTHH-MM-SS__episode{episode}__{title}.pkl

        :param path_folder: Path to folder
        :param vehicle: Vehicle that is simulated
        :param step_size: Stepsize used in this simulation
        :param nu_c_init: water current information in the simulation (body frame) array 6x1
        :param shapes: Shapes for 3d Animation that were used (static)
        :param title: Optional title for addendum
        :param episode: Episode number
        :return: None
        """
        # Some variables for saving the file
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        self.file_save_name = os.path.join(path_folder, f"{utc_str}__episode{episode}__{title}.pkl")
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
            "nu_c": ArrayList(nu_c_init),
            "shapes": [deepcopy(shape) for shape in shapes],
            "episode": episode,
            "step_size": step_size
        }

    def update(self, nu_c: np.ndarray) -> None:
        """
        should be called in the end of each simulation step

        :param nu_c: water current at that time array 6x1
        :return: None
        """
        self.storage["vehicle"]["states"].add_row(self.vehicle.state)
        self.storage["vehicle"]["states_dot"].add_row(self.vehicle._state_dot)
        self.storage["vehicle"]["u"].add_row(self.vehicle.u)
        self.storage["nu_c"].add_row(nu_c)

    def save(self) -> str:
        """
        Function to save the pickle file

        :return: path to where file is saved
        """
        self.storage["vehicle"]["states"] = self.storage["vehicle"]["states"].get_nparray()
        self.storage["vehicle"]["states_dot"] = self.storage["vehicle"]["states_dot"].get_nparray()
        self.storage["vehicle"]["u"] = self.storage["vehicle"]["u"].get_nparray()

        with open(self.file_save_name, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.storage, outp, pickle.HIGHEST_PROTOCOL)

        return self.file_save_name

    def load(self, file_name: str) -> dict:
        """
        Function to load an existing storage

        :param file_name: path to file
        :return: the loaded storage
        """
        with open(file_name, 'rb') as inp:
            self.storage = pickle.load(inp)
            return self.storage

    @property
    def states(self):
        return self.storage["vehicle"]["states"][:]

    @property
    def positions(self) -> np.ndarray:
        r"""
        Returns ALL the position of the AUV in NED coordinates x, y, z

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"][:, 0:3]

    @property
    def attitudes(self) -> np.ndarray:
        r"""
        Returns ALL the attitude (euler angles) of the AUV wrt. to NED coordinates roll, pitch, yaw.

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"][:, 3:6]

    @property
    def step_size(self) -> float:
        return self.storage["step_size"]

    @property
    def u(self) -> np.ndarray:
        return self.storage["vehicle"]["u"][:]

    @property
    def nu_c(self) -> np.ndarray:
        return self.storage["nu_c"][:]
