import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property

from typing import List


class Shape(ABC):
    """
    This is a base class for any shape, should always contain center coordinates of position.
    """

    def __init__(self, position: np.ndarray):
        self.position = np.array(position)

    @abstractmethod
    def get_plot_variables(self) -> List[np.ndarray]:
        """
        Function that returns the plot variables for the matplotlib axes.surface_plot() function

        :return: return list of 1d arrays for plotting
        """
        pass


class Sphere(Shape):
    """
    Represents a sphere
    """

    def __init__(self, position: np.ndarray, radius: float):
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius

    def get_plot_variables(self):
        x_c, y_c, z_c = self.get_plot_shape(self.radius)
        return [[self.position[0] + x_c,
                self.position[1] + y_c,
                self.position[2] + z_c]]

    @staticmethod
    def get_plot_shape(radius: float, scale: int = 1, sweep1: int = 20, sweep2: int = 20):
        """
        Also used by capsule to create half spheres, therefor static method

        :param radius: radius of sphere
        :param scale: [0, 2] range for circle
        :param sweep1: first sweep of mesh
        :param sweep2: second sweep of mesh
        :return: x, y, z coordinates for plotting function
        """
        u, v = np.mgrid[0:scale * np.pi:sweep1*1j, 0:2*np.pi:sweep2*1j]
        x_c = radius * np.cos(u) * np.sin(v)
        y_c = radius * np.sin(u) * np.sin(v)
        z_c = radius * np.cos(v)
        return x_c, y_c, z_c


class Capsule(Shape):
    """
    Represents a Capsule, height is the total height, position is the center of the cylinder

    .. note:

        so far the half sphere we use a full sphere yet as long as there is no method for just plotting the
        necessary half sphere

    """

    def __init__(self, position: np.ndarray, radius: float, vec_top: np.ndarray):
        """

        :param position: Position of center of capsule
        :param radius: radius of capsule
        :param vec_top: line endpoint of axis of capsule (not to the very end, until sphere planar area)
        """
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius
        self.vec_top = vec_top
        self.vec_bot = self.position - (self.vec_top - self.position)

    def get_plot_variables(self):
        x_c, y_c, z_c = self.get_plot_shape_cyl
        return [[x_c, y_c, z_c], *self.get_plot_shape_sph]

    @cached_property
    def get_plot_shape_sph(self):
        x_c1, y_c1, z_c1 = Sphere.get_plot_shape(self.radius)
        x_c2, y_c2, z_c2 = [x_c1 + self.vec_top[0], y_c1 + self.vec_top[1], z_c1 + self.vec_top[2]]
        x_c1, y_c1, z_c1 = [x_c1 + self.vec_bot[0], y_c1 + self.vec_bot[1], z_c1 + self.vec_bot[2]]
        return [[x_c1, y_c1, z_c1], [x_c2, y_c2, z_c2]]

    @cached_property
    def get_plot_shape_cyl(self):
        """
        Adapted from:
        https://stackoverflow.com/questions/39822480/plotting-a-solid-cylinder-centered-on-a-plane-in-matplotlib
        """
        # vector in direction of axis
        v = self.vec_top - self.vec_bot

        # find magnitude of vector
        mag = np.linalg.norm(v)

        # unit vector in direction of axis
        v = v / mag

        # make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1)

        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 20)
        rsample = np.linspace(0, self.radius, 2)

        # use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)

        # rsample, theta = np.meshgrid(rsample, theta)

        # generate coordinates for surface
        # "Tube"
        x_c, y_c, z_c = [self.vec_bot[i] + v[i] * t + self.radius * np.sin(theta2) * n1[i] + self.radius * np.cos(theta2) * n2[i] for i in [0, 1, 2]]

        return x_c, y_c, z_c


def collision_sphere_sphere(pos1: np.ndarray, rad1: float, pos2: np.ndarray, rad2: float) -> bool:
    """
    Determining whether two sphere objects collide

    :param pos1: (3,) array for position of first object
    :param rad1: radius of first object
    :param pos2: (3,) array for position of second object
    :param rad2: radius of second object
    :return: returns true for collision
    """
    return np.linalg.norm(pos1 - pos2) <= rad1 + rad2


def collision_capsule_sphere(pos1: np.ndarray, h1: float, rad1: float, pos2: np.ndarray, rad2: float) -> bool:
    """
    Determining whether a cylinder collides with a sphere

    :param pos1: (3,) array for position of cylinder
    :param h1: height (total) of cylinder
    :param rad1: radius of cylinder
    :param pos2: (3,) array for position of sphere
    :param rad2: radius of sphere
    :return: returns true for collision
    """

