import numpy as np
import os
from functools import cached_property
from ..auvsim import AUVSim
from ..statespace import StateSpace


class BlueROV2(AUVSim):
    """
    The BlueROV2 in Heavy configuration (8 T200 Thruster) by Blue Robotics, is capable of depths up to 100 metres.
    More information on the BlueROV2 can be found here: https://bluerobotics.com/store/rov/bluerov2/

    This class is separated and written in 2 different version for the control matrix:
    - "direct":
    - "joystick"

    The parameters for the BlueROV2 are loaded via a xml file, SI units are used.
    The system identification data of the BlueROV2 are publicly available from
    - [1] Einarsson, Emil Már, and Andris Lipenitis. n.d. “Model Predictive Control for the BlueROV2,”
    - [2] Wu, Chu-Jou, and B Eng. n.d. “6-DoF Modelling and Control of a Remotely Operated Vehicle,”

    :param xml_path: Path to xml file with parameters for BlueROV2
    :param control_mode: Identifier for B Matrix and u_bound ["joystick", "direct"]

    """

    def __init__(self, xml_path: str = os.path.join(os.path.dirname(__file__), 'BlueROV2.xml'),
                 control_mode: str = "joystick"):
        super().__init__()  # Call inherited init functions and then add to it
        StateSpace.read_phys_para_from_xml(self, xml_path)  # Assign BlueROV2 parameters

        # These are the values for controlling the BlueROV2 in a simplified way via the joystick, assuming we have a
        # mapping from the six controls to 6dof movement
        if control_mode == 'joystick':
            self.K_thrust = 20  # Reduced maximum thrust here as from [2] for restricting too fast movement
            # B Matrix calculated from assumption in direct control mode with low level control
            self._B = np.array([
                [2.83, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.83, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.436, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.24, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.378]
            ]) * self.K_thrust
            self._u_bound = np.array([
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1]])
        # from C.-J. Wu and B. Eng, “6-DoF Modelling and Control of a Remotely Operated Vehicle,” p. 39.ff
        elif control_mode == 'direct':
            self.K_thrust = np.diag([40, 40, 40, 40, 40, 40, 40, 40])  # since each thruster is the same
            self.T_thrust = np.array([
                [0.707, 0.707, -0.707, -0.707, 0, 0, 0, 0],
                [-0.707, 0.707, -0.707, 0.707, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, -1, -1, -1],
                [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                [0.06, 0.06, -0.06, -0.06, 0.120, -0.120, 0.120, -0.120],
                [-0.189, 0.189, 0.189, -0.189, 0, 0, 0, 0]
            ])
            self._B = np.dot(self.T_thrust, self.K_thrust)
            self._u_bound = np.array([
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1]])
        else:
            raise KeyError("Invalid control mode for BlueROV2 initialization.")

    def B(self, nu) -> np.ndarray:
        return self._B

    @cached_property
    def u_bound(self) -> np.ndarray:
        return self._u_bound

    # These functions below are only needed for testing to enter scenarios (cached property does not come with a setter)
    def set_B(self, value):
        self._B = value

    def set_u_bound(self, value):
        self._u_bound = value
