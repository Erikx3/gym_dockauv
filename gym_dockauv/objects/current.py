import numpy as np

from ..utils import geomutils as geom


class Current:
    r"""
    Ocean current with constant alpha and beta and first order gauss markov process (linear state model) for simulation.

    `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, Chapter 8

    :param mu: Constant in
    :param V_min: Lower boundary for current speed
    :param V_max: upper boundary for current speed
    :param Vc_init: Initial current speed
    :param alpha_init: Initial :math:`\alpha` angle in rad
    :param beta_init: Initial :math:`\beta` angle in rad
    :param white_noise_std: Standard deviation :math:`\sigma` of the white noise
    """

    def __init__(self, mu: float, V_min: float, V_max: float, Vc_init: float, alpha_init: float, beta_init: float,
                 white_noise_std: float):

        self.mu = mu
        self.Vmin = V_min
        self.Vmax = V_max
        self.Vc = Vc_init
        self.alpha = alpha_init
        self.beta = beta_init
        self.white_noise_std = white_noise_std

    def __call__(self, Theta: np.ndarray) -> np.ndarray:
        r"""
        Returns the current velocity :math:`\boldsymbol{\nu}_c` in {b} for the AUV when called

        .. math ::

            \boldsymbol{v}_c^b = \boldsymbol{R}_b^n(\boldsymbol{\Theta_{nb}})^T \boldsymbol{v}_c^n

        :param Theta: Euler Angles array 3x1 :math:`[\phi, \theta, \psi]^T`
        :return: 6x1 array
        """
        phi = Theta[3]
        theta = Theta[4]
        psi = Theta[5]

        vel_current_NED = self.get_current_NED()
        vel_current_BODY = np.transpose(geom.Rzyx(phi, theta, psi)).dot(vel_current_NED)

        nu_c = np.array([*vel_current_BODY, 0, 0, 0])

        return nu_c

    def get_current_NED(self) -> np.ndarray:
        r"""
        Returns current in NED coordinates

        .. note:: The :math:`\alpha` and :math:`\beta` angles from initialization are assumed to be constant in NED,
            thus transformation from FLOW coordinate system to NED can be done here with varying :math:`V_c`

        .. math::

            \boldsymbol{v}_c = V_c \begin{bmatrix}
                \cos \alpha_c \\
                \sin \beta_c \\
                \sin \alpha_c cos \beta_c
            \end{bmatrix}^T

        :return: 3x1 array
        """
        vel_current_NED = np.array([self.Vc * np.cos(self.alpha) * np.cos(self.beta),
                                    self.Vc * np.sin(self.beta),
                                    self.Vc * np.sin(self.alpha) * np.cos(self.beta)])

        return vel_current_NED

    def sim(self, h: float) -> None:
        r"""
        Simulate one time step of the current dynamics according to linear state model

        .. math::

            \dot{V}_c + \mu V_c = w

        :param h: time step
        :return: None
        """
        w = np.random.normal(0, self.white_noise_std)
        if self.Vc >= self.Vmax and w >= 0 or self.Vc <= self.Vmin and w <= 0:
            Vc_dot = 0
        else:
            Vc_dot = -self.mu * self.Vc + w
        self.Vc += Vc_dot * h
