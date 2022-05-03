import numpy as np
import os
from functools import cached_property
from ..auvsim import AUVSim
from ..statespace import StateSpace

XML_PATH = os.path.join('LAUV.xml')  # Use os.path.join for ensuring cross-platform stability


class LAUV(AUVSim):
    """
    LAUV vehicle as used in https://github.com/simentha/gym-auv, described in his work:

     S. T. Havenstrøm, “From Beginner to Expert: Deep Reinforcement Learning Controller for 3D Path Following and
     Collision Avoidance by Autonomous Underwater Vehicles,” 2020, Accessed: Mar. 09, 2022. [Online]. Available:
     https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2780864

     Adapted from:

     J. E. D. Silva, B. Terra, R. Martins, and J. B. D. Sousa, “Modeling and Simulation of the LAUV Autonomous
     Underwater Vehicle"

    With help of formulas from:

    T. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control. 2011. doi: 10.1002/9781119994138.

    """

    def __init__(self, xml_path: str = os.path.join(os.path.dirname(__file__), 'LAUV.xml')):
        super().__init__()
        # Additional introduced variables in this class!
        self.N_urf = 0.0
        self.N_uvf = 0.0
        self.N_uvb = 0.0
        self.M_uqf = 0.0
        self.M_uwf = 0.0
        self.M_uwb = 0.0
        self.Z_uqf = 0.0
        self.Z_uwf = 0.0
        self.Z_uwb = 0.0
        self.Y_urf = 0.0
        self.Y_uvf = 0.0
        self.Y_uvb = 0.0
        self.N_vv = 0.0
        self.M_ww = 0.0
        self.Z_qq = 0.0
        self.Y_rr = 0.0
        self.N_v = 0.0
        self.M_w = 0.0
        self.Z_q = 0.0
        self.Y_r = 0.0
        self.N_uudr = 0.0
        self.M_uuds = 0.0
        self.Z_uuds = 0.0
        self.Y_uudr = 0.0
        # Load state space variables from xml
        StateSpace.read_phys_para_from_xml(self, xml_path)

    def B(self, nu: np.ndarray) -> np.ndarray:
        u = nu[0]
        B = np.array([[1, 0, 0],
                      [0, self.Y_uudr * (u ** 2), 0],
                      [0, 0, self.Z_uuds * (u ** 2)],
                      [0, 0, 0],
                      [0, 0, self.M_uuds * (u ** 2)],
                      [0, self.N_uudr * (u ** 2), 0]])
        return B

    def D(self, nu: np.ndarray):
        """
        Overwrite base class function for damping matrix D, since this model is more complex

        :param nu: relativ body frame speed
        :return: 6x6 array
        """
        u = abs(nu[0])
        v = abs(nu[1])
        w = abs(nu[2])
        p = abs(nu[3])
        q = abs(nu[4])
        r = abs(nu[5])

        D = -np.array([[self.X_u, 0, 0, 0, 0, 0],
                       [0, self.Y_v, 0, 0, 0, self.Y_r],
                       [0, 0, self.Z_w, 0, self.Z_q, 0],
                       [0, 0, 0, self.K_p, 0, 0],
                       [0, 0, self.M_w, 0, self.M_q, 0],
                       [0, self.N_v, 0, 0, 0, self.N_r]])
        D_n = -np.array([[self.X_uu * u, 0, 0, 0, 0, 0],
                         [0, self.Y_vv * v, 0, 0, 0, self.Y_rr * r],
                         [0, 0, self.Z_ww * w, 0, self.Z_qq * q, 0],
                         [0, 0, 0, self.K_pp * p, 0, 0],
                         [0, 0, self.M_ww * w, 0, self.M_qq * q, 0],
                         [0, self.N_vv * v, 0, 0, 0, self.N_rr * r]])
        L = -np.array([[0, 0, 0, 0, 0, 0],
                       [0, self.Y_uvb + self.Y_uvf, 0, 0, 0, self.Y_urf],
                       [0, 0, self.Z_uwb + self.Z_uwf, 0, self.Z_uqf, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, self.M_uwb + self.M_uwf, 0, self.M_uqf, 0],
                       [0, self.N_uvb + self.N_uvf, 0, 0, 0, self.N_urf]])
        return D + D_n + L * u

    @cached_property
    def u_bound(self) -> np.ndarray:
        u_bound = np.array([
            [0, 14],
            [-30 * np.pi / 180, 30 * np.pi / 180],
            [-30 * np.pi / 180, 30 * np.pi / 180]
        ])
        return u_bound
