import numpy as np
from functools import cached_property


class LowPassFilter:
    """
    Provide Low pass Filter calculations
    """
    def __init__(self, T1=0.2, sample_time=1):
        self.T1 = T1  # Time constant in [s] for low-pass filter
        self.sample_time = sample_time # E.g. equals the step_size

    @cached_property
    def alpha(self) -> float:
        r"""
        Smoothing factor for discrete time implementation, :math:`0 <= \alpha <= 1`. When :math:`\alpha=0.5`,
        the low pass time constant is equal to the sample time, if it is smaller than 0.5, the time constant is larger
        than then the sample time.

        .. math::

            \alpha = \frac{\Delta_T}{T_1 + \Delta_T}

        :return: smoothing factor
        """

        return self.sample_time / (self.sample_time + self.T1)

    def apply_lowpass(self, x: np.ndarray, y_prev: np.ndarray) -> np.ndarray:
        r"""
        Applies low-pass according to a discrete-time implementation of a simple low-pass filter:

        .. math::

            y_i = \alpha x_i + (1 - \alpha) * y_{i-1}

        :param x: e.g. command at time i
        :param y_prev: previous command at time i-1
        :return: resulting command at time i
        """
        y = self.alpha * x + (1 - self.alpha) * y_prev
        return y
