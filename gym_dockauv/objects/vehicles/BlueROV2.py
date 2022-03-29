import numpy as np
import os
from ..statespace import StateSpace

XML_PATH = os.path.join('BlueROV2.xml')  # Use os.path.join for ensuring cross-platform stability


class BlueROV2(StateSpace):
    """
    Some description about the BlueROV2 used here

    Also make note about loading in the data and where they re from for future easier reference
    """

    def __init__(self):
        super().__init__()  # Call inherited init functions and then add to it
        StateSpace.read_para_from_xml(self, XML_PATH)  # Assign BlueROV2 parameters

    def B(self) -> np.ndarray:
        # TODO: Adapt input of BlueROV2 (for now it applies direct force uncoupled in each direction, maybe do
        #  feasibility study also about control of thrusters, make combination etc
        B = np.diag(np.ones(6))
        return B


