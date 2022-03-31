import numpy as np


def ssa(angle: np.ndarray) -> np.ndarray:
    r"""
    Express input angle between :math:`-\pi` and :math:`+\pi` rad

    :param angle: input angle in rad
    :return: angle in rad
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def Rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    r"""
    Rotation matrix from {b} to {n}, to obtain a vector in NED coordinates that is expressed in the body frame,
    a matrix multiplication with the rotation matrix is used. For the reserve way, use the transposed rotation
    matrix. In the formula below abbreviations :math:`s` for sin and :math:`c` for cos are used.

    .. math::

        \boldsymbol{R}_b^n(\boldsymbol{\Theta}_{nb}) = \begin{bmatrix}
            c \psi c \theta & -s \psi c \phi + c \psi s \theta s \phi & s \psi s \phi + c \psi c \phi s \theta \\
            s \psi c \theta & c \psi c \phi + s \phi s \theta s \psi & -c \psi s \phi + s \theta s \psi c \phi \\
            -s \theta & c \theta s \phi & c \theta c \phi
        \end{bmatrix}

    :param phi: euler angle around the x axis
    :param theta: euler angle around the y axis
    :param psi: euler angle around the z axis
    :return: array 3x3
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth]),
        np.hstack([spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi]),
        np.hstack([-sth, cth * sphi, cth * cphi])])


def Tzyx(phi: float, theta: float) -> np.ndarray:
    r"""
    Angular transformation matrix from {b} to {n}. In the formula below abbreviations :math:`s` for sin, :math:`c`
    for cos and :math:`t` for tan are used.

    .. note:: Note that this transformation is not well-defined for :math:`\theta = \frac{\pi}{2}`, however we do not
        expect an AUV to do that. An alternative approach could be quaternions, but are not implemented here.

    .. math::

        \boldsymbol{T}_{\Theta}^n(\boldsymbol{\Theta}_{nb}) = \begin{bmatrix}
            1 & s \phi t \theta & c \phi t \theta \\
            0 & c \phi & -s \phi \\
            0 & \frac{s \phi}{c \theta} & \frac{c \phi}{c \theta}
        \end{bmatrix}

    :param phi: euler angle around the x axis
    :param theta: euler angle around the y axis
    :return: array 3x3
    """

    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.vstack([
        np.hstack([1, sphi * tth, cphi * tth]),
        np.hstack([0, cphi, -sphi]),
        np.hstack([0, sphi / cth, cphi / cth])])


def J(eta: np.ndarray) -> np.ndarray:
    r"""
    Transformation and rotation matrix combined for obtaining kinematic equation with :math:`\dot{\boldsymbol{\eta}}
    = \boldsymbol{J}_{\Theta}(\boldsymbol{\eta}) \boldsymbol{\nu}` and

    .. math::

        \boldsymbol{J}_{\Theta}(\boldsymbol{\eta}) = \begin{bmatrix}
            \boldsymbol{R}_b^n(\boldsymbol{\Theta}_{nb}) & 0\\
            0 & \boldsymbol{T}_{\Theta}(\boldsymbol{\Theta}_{nb})
        \end{bmatrix}

    :param eta: pose coordinates vector :math:`\boldsymbol{\eta} = [x \: y \: z \: \phi \: \theta \: \psi]^T`
    :return: array 6x6
    """
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta)
    zero = np.zeros((3, 3))

    return np.vstack([
        np.hstack([R, zero]),
        np.hstack([zero, T])])


def S_skew(a: np.ndarray) -> np.ndarray:
    r"""
    Returns symmetric skew matrix 3x3, where :math:`\boldsymbol{a} = [a_1, a_2, a_3]^T`:

    .. math::

        \boldsymbol{S}(\boldsymbol{a}) = \begin{bmatrix}
            0 & -a_3 & a_2 \\
            a_3 & 0 & -a_1 \\
            -a_2 & a_1 & 0
        \end{bmatrix}

    :param a: 3x1 input vector
    :return: array 3x3
    """
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    return np.vstack([
        np.hstack([0, -a3, a2]),
        np.hstack([a3, 0, -a1]),
        np.hstack([-a2, a1, 0])])


def _H(r: np.ndarray) -> np.ndarray:
    r"""
    Helper function to determine center of origin offset to center of gravity

    :param r: distance from origin
    :return: array 6x6
    """
    I3 = np.identity(3)
    zero = np.zeros((3, 3))

    return np.vstack([
        np.hstack([I3, np.transpose(S_skew(r))]),
        np.hstack([zero, I3])])


def move_to_CO(A_CG: np.ndarray, r_g: np.ndarray) -> np.ndarray:
    """
    Function for e.g. the rigid body mass matrix to include the offset of the center of origin to the center of gravity

    :param A_CG: input matrix without offset
    :param r_g: distance from origin
    :return: array 6x6
    """
    H = _H(r_g)
    Ht = np.transpose(H)
    A_CO = Ht.dot(A_CG).dot(H)
    return A_CO
