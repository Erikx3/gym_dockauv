import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property

import gym_dockauv.utils.geomutils as geom


class StateSpace:
    """
    This class represents the Metaclass for a statespace. It basically serves as template for AUV dynamics with 6dof.
    The __init__ function can be overwritten by simply adding the derivatives of the state space, that are not zero.
    Do not forget to call super().__init__() in any child class __init__ constructor.
    """

    @abstractmethod
    def __init__(self):
        # General AUV parameters
        self.m = 0

        # Moments of Inertia
        self.I_x = 0
        self.I_y = 0
        self.I_z = 0
        self.I_xy = self.I_xz = self.I_yz = 0

        self.r_G = [0, 0, 0]  # [meters], x, y, z distance CG from CO (typically at CB)

    @cached_property
    def Ig(self) -> np.ndarray:
        """
        LOL

        :return: moment of inertia array 3x3
        """
        Ig = np.array([
            [self.I_x, -self.I_xy, -self.I_xz],
            [-self.I_xy, self.I_y, -self.I_yz],
            [self.I_xz, -self.I_yz, self.I_z]
        ])
        return Ig

    @cached_property
    def M_RB(self) -> np.ndarray:
        r"""
        This function builds the rigid body mass matrix

        .. math::

            M = \begin{bmatrix}
                    1 & 4 & 7 \\
                    2 & 5 & 8 \\
                    3 & 6 & 9
                \end{bmatrix}

        :return: array 6x6
        """
        M_RB_CG = np.vstack([
            np.hstack([self.m * np.identity(3), np.zeros(3)]),
            np.hstack([np.zeros(3), self.Ig])
        ])

        M_RB_CO = geom.move_to_CO(M_RB_CG, self.r_G)
        return M_RB_CO
