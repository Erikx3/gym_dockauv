import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property

from typing import List


class Shape(ABC):
    """
    This is a base class for any shape, should always contain coordinates of position.
    """

    def __init__(self, position: np.ndarray):
        self.position = np.array(position)

    @abstractmethod
    def get_plot_variables(self) -> List[np.ndarray]:
        """
        Function that returns the plot variables for the matplotlib axes.surface_plot function

        :return: return list of 1d arrays for plotting
        """
        pass


class Sphere(Shape):

    def __init__(self, position: np.ndarray, radius: float):
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius

    def get_plot_variables(self):
        x_c, y_c, z_c = self.get_plot_shape
        return [self.position[0] + x_c,
                self.position[1] + y_c,
                self.position[2] + z_c]

    @cached_property
    def get_plot_shape(self):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x_c = self.radius * np.cos(u) * np.sin(v)
        y_c = self.radius * np.sin(u) * np.sin(v)
        z_c = self.radius * np.cos(v)
        return x_c, y_c, z_c


class Cylinder(Shape):

    def __init__(self, position: np.ndarray, radius: float, height: float):
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius
        self.height = height

    def get_plot_variables(self):
        x_c, y_c, z_c = self.get_plot_shape
        return [self.position[0] + x_c,
                self.position[1] + y_c,
                self.position[2] + z_c]

    @cached_property
    def get_plot_shape(self):
        z = np.linspace(-self.height/2, self.height/2, 40)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_c = np.meshgrid(theta, z)
        x_c = self.radius * np.cos(theta_grid)
        y_c = self.radius * np.sin(theta_grid)
        return x_c, y_c, z_c
