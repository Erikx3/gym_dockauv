import numpy as np
import xml.etree.ElementTree as ET
from functools import cached_property
from abc import ABC, abstractmethod
from numpy.linalg import inv
from math import cos, sin

from ..utils import geomutils as geom


class StateSpace(ABC):
    r"""
    This class represents the baseclass for a statespace. It basically serves as template for AUV dynamics with 6dof.
    The __init__ function can be overwritten by simply adding the derivatives of the state space, that are not zero
    or by using the function for reading these values in via a xml file. Do not forget to call super().__init__() in
    any child class __init__ constructor. Look at the example vehicles in the directory "vehicles"

    The formula below is used as a description of the **kinetic** state space and retrieved by
    `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, page 188,
    Equations of Relative Motion with the assumption of an irrotational and constant (or very slow changing) current:

    .. math ::
        \underbrace{\boldsymbol{M}_{RB} \boldsymbol{\dot{\nu}}
        + \boldsymbol{C}_{RB}(\boldsymbol{\nu})\boldsymbol{\nu}}_{\text{rigid-body terms}}
        + \underbrace{\boldsymbol{M}_A \boldsymbol{\dot{\nu}}_r + C_A(\boldsymbol{\nu}_r) \boldsymbol{\nu}_r
        + D(\boldsymbol{\nu}_r) \boldsymbol{\nu}_r}_{\text{hydrodynamic terms}}
        + \underbrace{g(\eta)}_{\text{hydrostatic terms}} = \boldsymbol{\tau}

    Where the relative velocity results from the absolute velocity and current velocity :math:`\boldsymbol{\nu}_r
    = \boldsymbol{\nu} - \boldsymbol{\nu}_c`

    .. note:: Assumptions:

        - Lift forces are neglected here for speeds up to 2m/s, however they can be easily added to the damping term by
          creating matrix :math:`\boldsymbol{L}`.

        - The current is irrotational and constant in {n}

        - Values for the Inertia in this class can be entered as the origin at center of gravity, the transformation
          will be performed by this class


    .. note:: Child Class:

        The BlueROV2 is implemented and an example for another AUV dynamics implementation, where we also need more
        variables, is shown under **TODO**. We use multiple inheritance for the vehicles, since they share a lot of
        basic functionality.

        Make sure to call super().__init__() within the Child class initialization function, when you overwrite the
        base class init, and then add further needed variables to your class (for example using the xml parser).
        Further on make sure the initialized variables are in the correct type, e.g. do not write mass = None,
        since the xml parser inherit the variable type from the initialization
    """

    def __init__(self):
        # General AUV description
        self.name = "AUV_name_here"
        self.version = 0.0

        # General AUV parameters
        self.m = 0.0
        self.g = 9.81
        self.BY = 0.0  # Buoyancy in [N]

        # Moments of Inertia with origin at gravity! (We do the transformation later)
        self.I_x = 0.0
        self.I_y = 0.0
        self.I_z = 0.0
        self.I_xy = self.I_xz = self.I_yz = 0.0

        # [meters], x_G, y_G, z_G distance CG from CO (typically at CB)
        self.x_G = self.y_G = self.z_G = 0.0

        # [meters], x_B, y_B, z_B distance CB from CO (typically at CB, thus all zero)
        self.x_B = self.y_B = self.z_B = 0.0

        # Added Mass variables
        self.X_udot = self.Y_vdot = self.Z_wdot = self.K_pdot = self.M_qdot = self.N_rdot = 0.0

        # Linear Damping parameters
        self.X_u = self.Y_v = self.Z_w = self.K_p = self.M_q = self.N_r = 0.0

        # Quadratic Damping parameters
        self.X_uu = self.Y_vv = self.Z_ww = self.K_pp = self.M_qq = self.N_rr = 0.0

    @cached_property
    def W(self) -> float:
        return self.m * self.g

    @cached_property
    def I_g(self) -> np.ndarray:
        """
        Inertia Moments matrix about the body's center of gravity

        :return: array 3x3
        """
        Ig = np.array([
            [self.I_x, -self.I_xy, -self.I_xz],
            [-self.I_xy, self.I_y, -self.I_yz],
            [self.I_xz, -self.I_yz, self.I_z]
        ])
        return Ig

    @cached_property
    def I_b(self) -> np.ndarray:
        r"""
        Inertia Moments matrix about an arbitrary origin

        .. math::
            \boldsymbol{I}_b = \boldsymbol{I}_g - m \boldsymbol{S}^2(\boldsymbol{r}_g)

        :return: array 3x3
        """
        # Both calculation yield the same
        I_b = self.I_g + self.m * geom.S_skew(self.r_G).dot(geom.S_skew(self.r_G).T)
        # I_b = self.I_g - self.m * (self.r_G[:, None].dot(self.r_G[:, None].T) - self.r_G.T.dot(self.r_G) * np.identity(3))
        return I_b

    @cached_property
    def r_G(self) -> np.ndarray:
        r"""
        :math:`\boldsymbol{r}_G = [x_G \: y_G \: z_G]^T` distance CG from CO (typically at CB)

        :return: array 3x1
        """
        return np.array([self.x_G, self.y_G, self.z_G])

    @cached_property
    def r_B(self) -> np.ndarray:
        r"""
        :math:`\boldsymbol{r}_B = [x_B \: y_B \: z_B]^T` distance CB from CO (typically at CB, thus all zero)

        :return: array 3x1
        """
        return np.array([self.x_B, self.y_B, self.z_B])

    @cached_property
    def M_RB(self) -> np.ndarray:
        r"""
        Builds and returns the rigid body mass matrix as given in the formula below

        .. math::

            \boldsymbol{M}_{RB} = \begin{bmatrix}
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
            np.hstack([self.m * np.identity(3), np.zeros((3, 3))]),
            np.hstack([np.zeros((3, 3)), self.I_g])
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

            \boldsymbol{M}_A = \begin{bmatrix}
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
        Retrieves the total inverse of the mass matrices (which will end up on the RHS of the EOM)

        :return: array 6x6
        """
        M = self.M_RB + self.M_A
        return inv(M)

    def C_RB(self, nu_r: np.ndarray) -> np.ndarray:
        r"""
        The skew-symmetric cross-product operation on :math:`\boldsymbol{M}_{RB}` yields the rigid-body centripetal
        Coriolis matrix :math:`\boldsymbol{C}_{RB}`. We use the velocity-independent parametrization as in
        `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, page 55, or Havenstrøm,
        Simen Theie. 2020. “From Beginner to Expert: Deep Reinforcement Learning Controller for 3D Path Following and
        Collision Avoidance by Autonomous Underwater Vehicles.”
        https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2780864 p. 19. where :math:`\boldsymbol{\nu}_2` represents the
        angular velocity vector, :math:`S` the cross product operator and :math:`\boldsymbol{I}_b` is the inertia
        matrix about an arbitrary origin (p.51)

        .. math::

            \boldsymbol{C}_{RB} = \begin{bmatrix}
                m \boldsymbol{S}(\boldsymbol{\nu}_2) &
                -m \boldsymbol{S}(\boldsymbol{\nu}_2) \boldsymbol{S}(\boldsymbol{r}_G) \\
                m \boldsymbol{S}(\boldsymbol{r}_G) \boldsymbol{S}(\boldsymbol{\nu}_2) &
                -\boldsymbol{S}(\boldsymbol{I}_b \boldsymbol{\nu}_2)
            \end{bmatrix}

        :param nu_r: relative velocity vector :math:`\boldsymbol{\nu}_r = [u \: v \: w \: p \: q \: r]^T`
        :return: array 6x6
        """
        nu_2 = nu_r[3:6]

        C_RB_CO = np.vstack([
            np.hstack([self.m * geom.S_skew(nu_2), -self.m * geom.S_skew(nu_2).dot(geom.S_skew(self.r_G))]),
            np.hstack([self.m * geom.S_skew(self.r_G).dot(geom.S_skew(nu_2)), -geom.S_skew(self.I_b.dot(nu_2))])
        ])
        return C_RB_CO

    def C_A(self, nu_r: np.ndarray) -> np.ndarray:
        r"""
        The skew-symmetric cross-product operation on :math:`\boldsymbol{M}_{A}` yields the added mass centripetal
        Coriolis matrix :math:`\boldsymbol{C}_{A}`. Below we used the most generic form, so we are able to adapt to
        changes someone made to :math:`\boldsymbol{M}_{A}`


        .. math::

            \boldsymbol{C}_{A} = \begin{bmatrix}
                \boldsymbol{0}_{3 \times 3} &
                -\boldsymbol{S}(\boldsymbol{M}_{A,11} \boldsymbol{\nu}_1 + \boldsymbol{M}_{A,12} \boldsymbol{\nu}_2) \\
                -\boldsymbol{S}(\boldsymbol{M}_{A,11} \boldsymbol{\nu}_1 + \boldsymbol{M}_{A,12} \boldsymbol{\nu}_2) &
                -\boldsymbol{S}(\boldsymbol{M}_{A,21} \boldsymbol{\nu}_1 + \boldsymbol{M}_{A,22} \boldsymbol{\nu}_2)
            \end{bmatrix}

        In the case of a diagonal added mass matrix :math:`\boldsymbol{M}_{A}`, this leads to the following result
        (Fossen 2011, p. 121)

        .. math::

            \boldsymbol{C}_{A} = \begin{bmatrix}
                0 & 0 & 0 & 0 & -Z_{\dot{w}} w & Y_{\dot{v}} v \\
                0 & 0 & 0 & Z_{\dot{w}} w & 0 & -X_{\dot{u}} u \\
                0 & 0 & 0 & -Y_{\dot{v}} v & X_{\dot{u}} u & 0 \\
                0 & -Z_{\dot{w}} w & Y_{\dot{v}} v & 0 & -N_{\dot{r}} r & M_{\dot{q}} q \\
                Z_{\dot{w}} w & 0 & -X_{\dot{u}} u & N_{\dot{r}} r & 0 & -K_{\dot{p}} p \\
                -Y_{\dot{v}} v & X_{\dot{u}} u & 0 & -M_{\dot{q}} q & K_{\dot{p}} p & 0 \\
            \end{bmatrix}

        :param nu_r: relative velocity vector :math:`\boldsymbol{\nu}_r = [u \: v \: w \: p \: q \: r]^T`
        :return: array 6x6
        """

        nu_1 = nu_r[0:3]
        nu_2 = nu_r[3:6]
        M_A11 = self.M_A[0:3, 0:3]
        M_A12 = self.M_A[0:3, 3:6]
        M_A21 = self.M_A[3:6, 0:3]
        M_A22 = self.M_A[3:6, 3:6]

        C_A_CO = np.vstack([
            np.hstack([np.zeros((3, 3)), -geom.S_skew(M_A11 @ nu_1 + M_A12 @ nu_2)]),
            np.hstack([-geom.S_skew(M_A11 @ nu_1 + M_A12 @ nu_2), -geom.S_skew(M_A21 @ nu_1 + M_A22 @ nu_2)])
        ])

        return C_A_CO

    def C(self, nu_r: np.ndarray) -> np.ndarray:
        r"""
        Returns the total of the coriolis matrices

        :param nu_r: relative velocity vector :math:`\boldsymbol{\nu}_r = [u \: v \: w \: p \: q \: r]^T`
        :return: array 6x6
        """
        C = self.C_RB(nu_r) + self.C_A(nu_r)
        return C

    def D(self, nu_r: np.ndarray) -> np.ndarray:
        r"""
        hydrodynamic damping modelled in linear viscous damping and quadratic damping.

        .. note:: Here we assume again decoupling such that the linear and nonlinear damping matrices are given as below

        .. math::

            \boldsymbol{D}(\boldsymbol{\nu}_r) = \boldsymbol{D}_L + \boldsymbol{D}_{NL}(\boldsymbol{\nu}_r)

        Linear Damping Matrix

        .. math::

            \boldsymbol{D}_L = \begin{bmatrix}
                X_u & 0 & 0 & 0 & 0 & 0 \\
                0 & Y_v & 0 & 0 & 0 & 0 \\
                0 & 0 & Z_w & 0 & 0 & 0 \\
                0 & 0 & 0 & K_p & 0 & 0 \\
                0 & 0 & 0 & 0 & M_q & 0 \\
                0 & 0 & 0 & 0 & 0 & N_r
            \end{bmatrix}

        Nonlinear (quadratic) damping matrix

        .. math::

            \boldsymbol{D}_{NL} = \begin{bmatrix}
                X_{u|u|} |u| & 0 & 0 & 0 & 0 & 0 \\
                0 & Y_{v|v|} |v| & 0 & 0 & 0 & 0 \\
                0 & 0 & Z_{w|w|} |w| & 0 & 0 & 0 \\
                0 & 0 & 0 & K_{p|p|} |p| & 0 & 0 \\
                0 & 0 & 0 & 0 & M_{q|q|} |q| & 0 \\
                0 & 0 & 0 & 0 & 0 & N_{r|r|} |r|
            \end{bmatrix}

        :param nu_r: relative velocity vector :math:`\boldsymbol{\nu}_r = [u \: v \: w \: p \: q \: r]^T`
        :return: array 6x6
        """
        u = abs(nu_r[0])
        v = abs(nu_r[1])
        w = abs(nu_r[2])
        p = abs(nu_r[3])
        q = abs(nu_r[4])
        r = abs(nu_r[5])

        # Minus sign check: Necessary, since Derivatives are negative, then when on RHS of equation, there is another
        # minus sign, thus we need this minus sign so the drag counteracts actual movement, is also defined in Fossen
        # like that
        D_L = -np.array([[self.X_u, 0, 0, 0, 0, 0],
                         [0, self.Y_v, 0, 0, 0, 0],
                         [0, 0, self.Z_w, 0, 0, 0],
                         [0, 0, 0, self.K_p, 0, 0],
                         [0, 0, 0, 0, self.M_q, 0],
                         [0, 0, 0, 0, 0, self.N_r]])

        D_NL = -np.array([[self.X_uu * u, 0, 0, 0, 0, 0],
                          [0, self.Y_vv * v, 0, 0, 0, 0],
                          [0, 0, self.Z_ww * w, 0, 0, 0],
                          [0, 0, 0, self.K_pp * p, 0, 0],
                          [0, 0, 0, 0, self.M_qq * q, 0],
                          [0, 0, 0, 0, 0, self.N_rr * r]])

        return D_L + D_NL

    def G(self, eta: np.ndarray) -> np.ndarray:
        r"""
        returns the restoring forces acting on the AUV

        .. math::

            \boldsymbol{G}(\boldsymbol{\eta}) = \begin{bmatrix}
                (W-B) \sin(\theta) \\
                - (W-B) \cos(\theta) \sin(\phi) \\
                - (W-B) \cos(\theta) \cos(\phi) \\
                - (y_G W - y_B B) \cos(\theta) \cos(\phi) + (z_G W - z_B B) \cos(\theta) \sin(\phi) \\
                (z_G W - z_B B) \sin(\theta) + (x_G W - x_B B) \cos(\theta) \cos(\phi) \\
                - (x_G W - x_B B) \cos(\theta) \sin(\phi) - (y_G W - y_B B) \sin(\theta)
            \end{bmatrix}

        .. note:: Assuming that the Center of Origin lies in the Center of Buoyancy and only :math:`z_g` is different
            from zero, this equation simplifies further to:

            .. math::
                \boldsymbol{G}(\boldsymbol{\eta}) = \begin{bmatrix}
                    (W-B) \sin(\theta) \\
                    - (W-B) \cos(\theta) \sin(\phi) \\
                    - (W-B) \cos(\theta) \cos(\phi) \\
                    z_G W \cos(\theta) \sin(\phi) \\
                    z_G W \sin(\theta)\\
                    0
                \end{bmatrix}

        :param eta: pose coordinates vector :math:`\boldsymbol{\eta} = [x \: y \: z \: \phi \: \theta \: \psi]^T`
        :return: 6x1 array
        """

        phi = eta[3]
        theta = eta[4]
        G = np.array([(self.W - self.BY) * sin(theta),
                      -(self.W - self.BY) * cos(theta) * sin(phi),
                      -(self.W - self.BY) * cos(theta) * cos(phi),
                      -(self.y_G * self.W - self.y_B * self.BY) * cos(theta) * cos(phi) + (
                              self.z_G * self.W - self.z_B * self.BY) * cos(theta) * sin(phi),
                      (self.z_G * self.W - self.z_B * self.BY) * sin(theta) + (
                              self.x_G * self.W - self.x_B * self.BY) * cos(theta) * cos(phi),
                      - (self.x_G * self.W - self.x_B * self.BY) * cos(theta) * sin(phi) - (
                              self.y_G * self.W - self.y_B * self.BY) * sin(theta)
                      ])
        return G

    @abstractmethod
    def B(self, nu) -> np.ndarray:
        r"""
        this function returns the control matrix :math:`\boldsymbol{B}` and is individual for each AUV and thus an
        abstract method

        When :math:`\boldsymbol{\tau}` represents the external forces and :math:`\boldsymbol{u}` the control input, then

        .. math::

            \boldsymbol{\tau} = \boldsymbol{B}_{6\times a} \boldsymbol{u}_{a\times 1}

        Where the dimension :math:`\boldsymbol{a}` is the number of actions available for the system
        :return: array 6xa
        """
        pass

    @property
    @abstractmethod
    def u_bound(self) -> np.ndarray:
        r"""
        This function shall return the upper and lower boundaries for the input u, where the first column represent
        the lower boundary.

        :return: array ax2
        """
        pass

    @staticmethod
    def read_phys_para_from_xml(obj, xml_path: str) -> None:
        r"""
        Parse flat xml with physical parameters for the vehicle and update them as an attribute to the instance. This
        function also checks, if the xml keys are already available as an attribute, otherwise this will throw an
        exception, since any Child classes from StateSpace should take care of any additional attributes in its init
        function

        :param obj: instance where attributes should be applied to
        :param xml_path: xml path to flat vehicle config
        :return: None
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for child in root:
            if hasattr(obj, child.tag):
                # Get the initialized attribute type of the instance
                setattr(obj, child.tag, type(getattr(obj, child.tag))(child.text))
            else:
                raise AttributeError("Bad and not allowed practice: Trying to parse xml data tag without it being "
                                     "initialized in init")
