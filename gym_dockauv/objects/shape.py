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
    def get_plot_variables(self) -> List[List[np.ndarray]]:
        """
        Function that returns the plot variables for the matplotlib axes.surface_plot() function

        :return: return list of list of arrays for plotting (one inner list contains plotting arrays)
        """
        pass


class Sphere(Shape):
    """
    Represents a sphere
    """

    def __init__(self, position: np.ndarray, radius: float):
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius

    def get_plot_variables(self) -> List[List[np.ndarray]]:
        x_c, y_c, z_c = self.get_plot_shape(self.radius)
        return [[self.position[0] + x_c,
                 self.position[1] + y_c,
                 self.position[2] + z_c]]

    @staticmethod
    def get_plot_shape(radius: float, scale: float = 1, sweep1: int = 20, sweep2: int = 20):
        """
        Also used by capsule to create half spheres, therefor static method

        :param radius: radius of sphere
        :param scale: [0, 1] range for circle
        :param sweep1: first sweep of mesh
        :param sweep2: second sweep of mesh
        :return: x, y, z coordinates for plotting function
        """
        u, v = np.mgrid[0:scale * np.pi:sweep1 * 1j, 0:2 * np.pi:sweep2 * 1j]
        x_c = radius * np.sin(u) * np.cos(v)
        y_c = radius * np.sin(u) * np.sin(v)
        z_c = radius * np.cos(u)
        return x_c, y_c, z_c


class Spheres:
    """
    Helper class to access and store data from all spheres

    Most important is to access all the positions and radius as a big array for vectorized functions

    This class can be enhanced with update or adding features, but are not needed in this case
    """

    def __init__(self, spheres: List[Sphere]):
        l = len(spheres)
        self.position = np.zeros((l, 3))
        self.radius = np.zeros(l)
        self.objs = []
        for count, sphere in enumerate(spheres):
            self.position[count, :] = sphere.position
            self.radius[count] = sphere.radius
            self.objs.append(sphere)

    def __call__(self) -> List[Sphere]:
        """
        When this class is called as a function, return the spheres

        :return: Return the list of spheres
        """
        return self.objs


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

    def get_plot_variables(self) -> List[List[np.ndarray]]:
        x_c, y_c, z_c = self.get_plot_shape_cyl
        return [[x_c, y_c, z_c], *self.get_plot_shape_sph]

    @cached_property
    def get_plot_shape_sph(self):
        # NOTE: This only works when capsule is aligned with z axis
        x_c1, y_c1, z_c1 = Sphere.get_plot_shape(self.radius, scale=0.5)
        x_c2, y_c2, z_c2 = [x_c1 + self.vec_top[0], y_c1 + self.vec_top[1], -z_c1 + self.vec_top[2]]
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
        x_c, y_c, z_c = [
            self.vec_bot[i] + v[i] * t + self.radius * np.sin(theta2) * n1[i] + self.radius * np.cos(theta2) * n2[i] for
            i in [0, 1, 2]]

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


def collision_sphere_spheres(pos1: np.ndarray, rad1: float, pos2: np.ndarray, rad2: np.ndarray) -> bool:
    """
    Determining whether one sphere 1 collides with any of multiple spheres

    :param pos1: (3,) array for position of first object
    :param rad1: radius of first object
    :param pos2: (n,3) array for position of all other spheres
    :param rad2: radius of all other spheres
    :return: returns true if collision
    """
    return np.any(np.linalg.norm(pos2 - pos1[None, :], axis=1) <= rad1 + rad2)


def collision_capsule_sphere(cap1: np.ndarray, cap2: np.ndarray, cap_rad: float,
                             sph_pos: np.ndarray, sph_rad: float) -> bool:
    """
    Determining whether a cylinder collides with a sphere

    :param cap1: (3,) array for the position of one of the capsule ends
    :param cap2: (3,) array for the position of the other capsule end
    :param cap_rad: radius of cylinder
    :param sph_pos: (3,) array for position of sphere
    :param sph_rad: radius of sphere
    :return: returns true for collision
    """
    # Closest distance between sphere center and capsule line
    dist = dist_line_point(sph_pos, cap1, cap2)
    # Check for collision
    return dist <= cap_rad + sph_rad


def intersec_dist_line_sphere(l1: np.ndarray, ld: np.ndarray, center: np.ndarray, rad: float):
    """
    From: https://iquilezles.org/articles/intersectors/


    :param l1: array (3,) for line starting point
    :param ld: array (3,) for the line direction from the starting point (does not need to be unit vector)
    :param center: array(3,) for the center of the sphere
    :param rad: radius of the sphere
    """

    oc = l1 - center  # type: np.ndarray
    rd = ld / np.linalg.norm(ld)
    b = np.dot(oc, rd)  # will be float
    c = np.dot(oc, oc) - rad * rad  # will be float
    h = b * b - c  # float
    if h < 0.0:
        return -np.inf  # no intersection
    h = np.sqrt(h)
    return min([-b + h, -b - h], key=abs)


def intersec_dist_lines_spheres_vectorized(l1: np.ndarray, ld: np.ndarray, center: np.ndarray, rad: np.ndarray):
    """
    Adapted from: https://iquilezles.org/articles/intersectors/

    This functions calculates the minimum distance of all intersection between a number of rays and multiple spheres

    nl is the number of rays, ns the number of sphere for the dimensions below in the description

    :param l1: array (nl, 3) for line starting points
    :param ld: array (nl, 3) for the line directions from the starting points (does not need to be unit vector)
    :param center: array(ns, 3) for the center of the spheres
    :param rad: array(ns,) for the radius of the spheres
    :return: array(nl,) with the shortest intersection distance in the direction of each ray, otherwise result is
        something negative (as no criteria for intersect "behind" the starting point is not well defined)
    """

    # array(3,) between each start point and center
    oc = l1[:, None] - center  # array(nl, ns, 3)
    rd = ld / np.linalg.norm(ld, axis=1)[:, None]  # array(nl, 3)
    # (nl, ns, 3) . (3, nl) -> (nl, ns, nl); then taking diagonal -> (nl, ns, 1)
    b = np.dot(oc, rd.T)[range(rd.shape[0]), :, range(rd.shape[0])] #np.diagonal(np.dot(oc, rd.T), axis1=1, axis2=2)
    # (nl, ns, 3) . (3, ns, nl) -> (nl, ns, nl); then taking diagonal -> (nl, ns, 1)
    c = np.linalg.norm(oc, axis=2)**2 - rad**2
    h = b * b - c  # float
    h[h < 0.0] = -np.inf  # no intersection at these points
    mask = h >= 0.0
    h[mask] = np.sqrt(h[mask])
    res = np.minimum(-b + h, -b - h)  # This would not work if starting point is within sphere
    # Only return the closest positive distance, otherwise it is just a random negative value (of 1st intersec)
    return res[np.arange(res.shape[0]), np.where(res > 0, res, np.inf).argmin(axis=1)]


def intersec_dist_line_capsule(l1: np.ndarray, ld: np.ndarray, cap1: np.ndarray, cap2: np.ndarray,
                               cap_rad: float) -> float:
    """
    return closest distance from starting point to intersection of capsule, otherwise returns -np.inf if no intersection
    is found. Intersection point can then be found by multiplying unit vector in direction of line by this distance.

    .. note::

        This solution ALWAYS finds the first intersection (if there is) in the direction of the ray vector. This
        means, it does not matter where the starting point is, it finds the first intersection in direction of the ray
        vector and can thus also return negative values

    Solution found here:
    https://iquilezles.org/articles/intersectors/

    :param l1: array (3,) for line starting point
    :param ld: array (3,) for the line direction from the starting point (does not need to be unit vector)
    :param cap1: array (3,) for capsule start
    :param cap2: array (3,) for capsule end
    :param cap_rad: capsule radius
    :return: distance from line starting point to intersection, -np.inf if no intersection at all
    """
    ba = cap2 - cap1
    oa = l1 - cap1
    # direction of vector as unit vector
    rd = ld / np.linalg.norm(ld)

    baba = np.dot(ba, ba)

    bard = np.dot(ba, rd)

    baoa = np.dot(ba, oa)

    rdoa = np.dot(rd, oa)

    oaoa = np.dot(oa, oa)

    a = baba - bard * bard

    b = baba * rdoa - baoa * bard

    c = baba * oaoa - baoa * baoa - cap_rad * cap_rad * baba

    h = b * b - a * c
    if h >= 0.0:
        t = (-b - np.sqrt(h)) / a
        y = baoa + t * bard
        # body
        if 0.0 < y < baba:
            return t
        # caps
        oc = oa if y <= 0.0 else l1 - cap2
        b = np.dot(rd, oc)
        c = np.dot(oc, oc) - cap_rad * cap_rad
        h2 = b * b - c
        if h2 > 0.0:
            return -b - np.sqrt(h2)
    return -np.inf


def intersec_dist_line_capsule_vectorized(l1: np.ndarray, ld: np.ndarray, cap1: np.ndarray, cap2: np.ndarray,
                                          cap_rad: float, default: float = -np.inf) -> np.ndarray:
    """
    Return the closest distance for multiple lines defined as in l1 and ld and find the shortest distances for ONE
    capsule

    :param l1: array (n,3) for lines starting point
    :param ld: array (n,3) for the lines direction from the starting point (does not need to be unit vector)
    :param cap1: array (3,) for capsule start
    :param cap2: array (3,) for capsule end
    :param cap_rad: capsule radius
    :param default: default number if no intersection is found
    :return: array(n,) with the distances calculated
    """
    ba = (cap2 - cap1)
    oa = l1 - cap1
    # direction of vector as unit vector
    rd = ld / np.linalg.norm(ld, axis=1)[:, None]

    baba = np.dot(ba, ba)

    bard = np.dot(rd, ba)

    baoa = np.dot(oa, ba)

    rdoa = np.diag(np.dot(rd, oa.T))

    oaoa = np.diag(np.dot(oa, oa.T))

    a = baba - bard * bard

    b = baba * rdoa - baoa * bard

    c = baba * oaoa - baoa * baoa - cap_rad * cap_rad * baba

    h = b * b - a * c

    res = np.zeros(l1.shape[0])

    # Vectorize conditional statements
    mask_h = h >= 0
    t = np.zeros(h.shape[0])
    t[~mask_h] = -np.inf
    t[mask_h] = (-b[mask_h] - np.sqrt(h[mask_h])) / a[mask_h]
    y = baoa + t * bard
    # body
    mask_body = (h >= 0) & (y > 0) & (y < baba)
    res[mask_body] = t[mask_body]

    # caps
    oc = np.zeros(l1.shape)
    oc[y <= 0.0] = oa[y <= 0.0]
    oc[y >= 0.0] = (l1 - cap2)[y >= 0.0]
    b = np.diag(np.dot(rd, oc.T))
    c = np.diag(np.dot(oc, oc.T)) - cap_rad * cap_rad

    h2 = b * b - c
    mask_caps = (h >= 0) & (h2 > 0.0) & ~mask_body

    res[mask_caps] = (-b[mask_caps] - np.sqrt(h2[mask_caps]))  # Double indexing to avoid runtime warning with sqrt

    # No intersection or behind:
    res[(h <= 0) | (res == 0)] = default
    return res


def dist_line_point(po: np.ndarray, l1: np.ndarray, l2: np.ndarray) -> float:
    """
    Function to calculate the closest distance between a line segment and a point

    From: https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python

    :param po: array (3,) for the point position
    :param l1: array (3,) for start of line
    :param l2: array (3,) for end of line
    :return: shortest distance between line and point
    """
    # normalized tangent vector
    d = np.divide(l2 - l1, np.linalg.norm(l2 - l1))

    # signed parallel distance components
    s = np.dot(l1 - po, d)
    t = np.dot(po - l2, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(po - l1, d)

    return np.hypot(h, np.linalg.norm(c))


def vec_line_point(po: np.ndarray, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """
    This function returns the vector pointing from the line towards the point

    :param po: array (3,) for the point position
    :param l1: array (3,) for start of line
    :param l2: array (3,) for end of line
    :return: array(3,) pointing from line to point
    """
    d_vec = (l2 - l1) / np.linalg.norm(l2 - l1)  # Unit vector for line
    v = po - l1
    t = np.dot(v, d_vec)  # Projection distance
    pro = l1 + t * d_vec  # Projected point on line
    return pro - po
