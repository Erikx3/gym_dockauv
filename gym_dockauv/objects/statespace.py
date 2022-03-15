import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property
from numpy.linalg import inv

import gym_dockauv.utils.geomutils as geom


class StateSpace:
    r"""
    This class represents the Parentclass for a statespace. It basically serves as template for AUV dynamics with
    6dof. The __init__ function can be overwritten by simply adding the derivatives of the state space, that are not
    zero or by using the function for reading it these in via a xml file. Do not forget to call super().__init__() in
    any child class __init__ constructor.

    The formula below is used as a description of the **kinetic** state space and retrieved by
    `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, page 188,
    Equations of Relative Motion with the assumption of irrotational current:

    .. math ::
        \underbrace{M_{RB} \dot{\nu} + C_{RB}(\nu)\nu}_{\text{rigid-body terms}}
        + \underbrace{M_A \dot{\nu}_r + C_A(\nu_r) \nu_r + D(\nu_r) \nu_r}_{\text{hydrodynamic terms}}
        + \underbrace{g(\eta)}_{\text{hydrostatic terms}} = \tau

    Where the relative velocity results from the absolute velocity and current velocity :math:`\nu_r = \nu - \nu_c`
    """

    # TODO: Think about doing this with xml file and setattr function later, solution for B matrix interesting
    @abstractmethod

    # TODO: !!! First choose solution for boldmath, either include package bm or add boldsymbol to all vectors nd matrices!!!
    def __init__(self):
        # General AUV parameters
        self.m = 0

        # Moments of Inertia
        self.I_x = 0
        self.I_y = 0
        self.I_z = 0
        self.I_xy = self.I_xz = self.I_yz = 0

        # [meters], x_G, y_G, z_G distance CG from CO (typically at CB)
        self.x_G = self.y_G = self.z_G = 0

        # Added Mass variables
        self.X_udot = self.Y_vdot = self.Z_wdot = self.K_pdot = self.M_qdot = self.N_rdot = 0

    @cached_property
    def I_g(self) -> np.ndarray:
        """
        Inertia Moments matrix

        :return: array 3x3
        """
        Ig = np.array([
            [self.I_x, -self.I_xy, -self.I_xz],
            [-self.I_xy, self.I_y, -self.I_yz],
            [self.I_xz, -self.I_yz, self.I_z]
        ])
        return Ig

    @cached_property
    def r_G(self) -> np.ndarray:
        r"""
        :math:`r_G = [x_G \: y_G \: z_G]^T` distance CG from CO (typically at CB)

        :return: array 3x1
        """
        return np.array([self.x_G, self.y_G, self.z_G])

    @cached_property
    def M_RB(self) -> np.ndarray:
        r"""
        Builds and returns the rigid body mass matrix as given in the formula below

        .. math::

            M_{RB} = \begin{bmatrix}
                    m & 0 & 0 & 0 & m z_G & -m y_G \\
                    0 & m & 0 & -m z_G & 0 & m x_G \\
                    0 & 0 & m & m y_G & -m x_G & 0 \\
                    0 & -m z_G & m y_G & I_x & -I_{xy} & -I_{xz} \\
                    m z_G & 0 & -m x_G & -I_{yx} & I_y & -I_{yz} \\
                    -m y_G & m x_G & 0 & -I_{zx} & -I_{zy} & I_z
                \end{bmatrix}

        :return: array 6x6
        """
        M_RB_CG = np.vstack([
            np.hstack([self.m * np.identity(3), np.zeros(3)]),
            np.hstack([np.zeros(3), self.I_g])
        ])

        M_RB_CO = geom.move_to_CO(M_RB_CG, self.r_G)
        return M_RB_CO

    @cached_property
    def M_A(self) -> np.ndarray:
        r"""
        Hydrodynamic term, added mass Matrix. For example the hydrodynamic derivative :math:`X_{\dot{u}}` means the
        hydrodynamic added mass force X in x direction due to acceleration on the x-axis (surge).

        .. note:: For most practical applications, the off diagonal terms of :math:`M_A` are negligible in comparison
            to the diagonal ones, thus the implementation here is kept simple w.r.t to a diagonal matrix. If you need
            the full added mass Matrix, simply overwrite this method in your child class

        .. math::

            M_A = \begin{bmatrix}
                    X_{\dot{u}} & 0 & 0 & 0 & 0 & 0 \\
                    0 & Y_{\dot{v}} & 0 & 0 & 0 & 0 \\
                    0 & 0 & Z_{\dot{w}} & 0 & 0 & 0 \\
                    0 & 0 & 0 & K_{\dot{p}} & 0 & 0 \\
                    0 & 0 & 0 & 0 & M_{\dot{q}} & 0 \\
                    0 & 0 & 0 & 0 & 0 & N_{\dot{r}}
                \end{bmatrix}

        :return: array 6x6
        """
        M_A = -np.diag([self.X_udot, self.Y_vdot, self.Z_wdot, self.K_pdot, self.M_qdot, self.N_rdot])
        return M_A

    @cached_property
    def M_inv(self) -> np.ndarray:
        """
        Retrieves the total inverse of the mass matrices (which will end up on the RHS)

        :return: array 6x6
        """
        M = self.M_RB + self.M_A
        return inv(M)

    def C_RB(self, nu_r: np.ndarray) -> np.ndarray:
        r"""
        The skew-symmetric cross-product operation on :math:`M_{RB}` yields the rigid-body centripetal Coriolis
        matrix :math:`C_{RB}`. We use the velocity-independent parametrization as in
        `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, page 55, where :math:`\nu_2`
        represents the angular velocity vector and :math:`S` the cross product operator

        .. math::

            \boldsymbol{C}_{RB} = \begin{bmatrix}
                    m S(\nu2) & -m S(\nu_2) S(r_G) \\
                    m S(r_G) S(\nu_2) & -S(I_b \nu_2)
                \end{bmatrix}

        :param nu_r: relative velocity vector :math:`\nu_r = [u \: v \: w \: p \: q \: r]^T`
        :return: array 6x6
        """
        nu_2 = nu_r[3:6]

        Ib = self.I_g - self.m * geom.S_skew(self.r_G).dot(geom.S_skew(self.r_G))

        C_RB_CO = np.vstack([
            np.hstack([self.m * geom.S_skew(nu_2), -self.m * geom.S_skew(nu_2).dot(geom.S_skew(self.r_G))]),
            np.hstack([self.m * geom.S_skew(self.r_G).dot(geom.S_skew(nu_2)), -geom.S_skew(Ib.dot(nu_2))])
        ])
        return C_RB_CO

# TODO Add the reduced matrices in the Bluerov subclass description
