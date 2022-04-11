import pickle
from copy import deepcopy

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


class EpisodeDataStorage:
    """
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
    }
    """

    def __init__(self, filename: str, vehicle: AUVSim, shapes: List[Shape]):
        # Some variables for saving the file
        self.filename = filename  # Should be formatted YYYY-MM-DD__HH-MM-SS__TEXT.pkl and match logger file name
        self.vehicle = vehicle  # Vehicle instance (not a copy, automatically a reference)
        self.storage = {
            "vehicle": {
                "object": vehicle,
                "states": [deepcopy(self.vehicle.state)],
                "states_dot": [deepcopy(self.vehicle._state_dot)],
                "u": [deepcopy(self.vehicle.u)],
                "shapes": [deepcopy(shape) for shape in shapes]
            }
        }

    def update(self) -> None:
        """
        should be called in the end of each simulation step

        :return: None
        """
        self.storage["vehicle"]["states"].append(deepcopy(self.vehicle.state))
        self.storage["vehicle"]["states_dot"].append(deepcopy(self.vehicle._state_dot))
        self.storage["vehicle"]["u"].append(deepcopy(self.vehicle.u))

    def save(self) -> None:
        """
        Function to save the pickle file
        """
        with open(self.filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.storage, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str):
        with open(file_name, 'rb') as inp:
            load_file = pickle.load(inp)
            return load_file
