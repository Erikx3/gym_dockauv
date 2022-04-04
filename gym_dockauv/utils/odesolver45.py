from typing import Callable
import numpy as np


def odesolver45(f: Callable, t: float, y: np.ndarray, h: float, *args, **kwargs):
    """
    Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.

    Adapted from here: https://github.com/simentha/gym-auv/blob/master/gym_auv/objects/auv3d.py

    :param f: functions RHS
    :param t: time (dummy here)
    :param y: state vector
    :param h: step size (fixed)
    :return: 4th and 5th order approximation results
    """
    s1 = f(t, y, *args, **kwargs)
    s2 = f(t, y + h * s1 / 4.0, *args, **kwargs)
    s3 = f(t, y + 3.0 * h * s1 / 32.0 + 9.0 * h * s2 / 32.0, *args, **kwargs)
    s4 = f(t, y + 1932.0 * h * s1 / 2197.0 - 7200.0 * h * s2 / 2197.0 + 7296.0 * h * s3 / 2197.0, *args, **kwargs)
    s5 = f(t, y + 439.0 * h * s1 / 216.0 - 8.0 * h * s2 + 3680.0 * h * s3 / 513.0 - 845.0 * h * s4 / 4104.0, *args, **kwargs)
    s6 = f(t,
           y - 8.0 * h * s1 / 27.0 + 2 * h * s2 - 3544.0 * h * s3 / 2565 + 1859.0 * h * s4 / 4104.0 - 11.0 * h * s5 / 40.0,
           *args, **kwargs)
    w = y + h * (25.0 * s1 / 216.0 + 1408.0 * s3 / 2565.0 + 2197.0 * s4 / 4104.0 - s5 / 5.0)
    q = y + h * (16.0 * s1 / 135.0 + 6656.0 * s3 / 12825.0 + 28561.0 * s4 / 56430.0 - 9.0 * s5 / 50.0 + 2.0 * s6 / 55.0)
    return w, q