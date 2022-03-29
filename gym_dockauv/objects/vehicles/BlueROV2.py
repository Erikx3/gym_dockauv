import numpy as np
import os
from functools import cached_property
from ..statespace import StateSpace

XML_PATH = os.path.join('BlueROV2.xml')  # Use os.path.join for ensuring cross-platform stability


class BlueROV2(StateSpace):
    """
    Some description about the BlueROV2 used here ...

    The parameters for the BlueROV2 are loaded via a xml file, SI units are used.
    The system identification data of the BlueROV2 are publicly available from ...
    """

    def __init__(self,  xml_path):
        super().__init__()  # Call inherited init functions and then add to it
        StateSpace.read_para_from_xml(self, xml_path)  # Assign BlueROV2 parameters

    @cached_property
    def B(self) -> np.ndarray:
        # TODO: Adapt input of BlueROV2 (for now it applies direct force uncoupled in each direction, maybe do
        #  feasibility study also about control of thrusters, make combination etc
        B = np.diag(np.ones(6))
        return B
