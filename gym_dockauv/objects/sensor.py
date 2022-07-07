import copy

import numpy as np
from functools import cached_property
from skimage.measure import block_reduce
from ..utils import geomutils as geom


class Radar:
    """
    Represent a radar sensor suite. Initialize with the alpha, beta view angle and the amount of the rays per degree

    Make sure alpha and beta is dividable by that amount, for now I throw an error if not.

    The rays are always defined in {b}, but to calculate the rays direction in {n} it simply takes one
    transformation, as long as the actual transformation matrix can be applied through the euler angles where the
    sensor is mounted on (function is provided)

    This implementation is done to support vectorized calculations.

    :param eta: position and euler angles
    :param freq: frequency of sensor suite updates
    :param alpha: range in vertical direction
    :param beta: range in horizontal direction
    :param ray_per_deg: number of rays per degree
    :param max_dist: maximum distance for all rays

    :var pos: array (3,) for the starting position of all rays
    :vartype pos: np.ndarray
    :var rd_b: array (n,3) for the direction of the rays in body frame, where n will be the total number of rays
        (does not change normal over the course of a simulation)
    :vartype rd_b: np.ndarray
    :var rd_n: array (n,3) for the direction of the rays in {n}
    :vartype rd_n: np.ndarray
    :var intersec_dist_fallback: array (n,) for saving the intersection distance for no intersection, the max distance
        will be set
    :vartype intersec_dist_fallback: np.ndarray
    :var end_pos_n: array (n,3) for saving the intersection point in {n}
    :vartype end_pos_n: np.ndarray

    """

    def __init__(self, eta: np.ndarray, freq: float, alpha: float = 2 * np.pi, beta: float = 2 * np.pi,
                 ray_per_deg: float = 5.0 * np.pi / 180, max_dist: float = 25, blocksize_reduce: int = 2):

        self.pos = eta[0:3]  # Initial starting position for all the rays, is always the same
        self.freq = freq
        self.max_dist = max_dist
        tol = 10e-8
        # Check for valid input, little quirky solution due to float precision
        if (alpha + tol) % ray_per_deg > 0.001 or (beta + tol) % ray_per_deg > 0.001:
            raise KeyError("Initialize the radar with valid ray_per_deg for alpha and beta.")
        self.alpha_max = alpha/2
        self.beta_max = beta/2
        # Create (n, 1) array for the alpha and beta angle of each array
        self.alpha = np.arange(-alpha / 2, alpha / 2 + tol, ray_per_deg)
        self.n_vertical = self.alpha.shape[0]  # Number of rays vertically
        self.alpha = np.repeat(self.alpha, repeats=(beta + tol) // ray_per_deg + 1, axis=0)
        self.beta = np.arange(-beta / 2, beta / 2 + tol, ray_per_deg)
        self.n_horizontal = self.beta.shape[0]  # Number of rays horizontally
        self.beta = np.tile(self.beta, (int((alpha + tol) // ray_per_deg + 1),))

        self.n_rays = self.alpha.shape[0]

        # Array (n, 3) of rays, where n will be the total number of rays
        self.rd_b = np.hstack([np.ones(self.n_rays)[:, None],
                               np.sin(self.beta)[:, None],
                               np.sin(self.alpha)[:, None]
                               ])
        # Normalize these vectors
        self.rd_b = self.rd_b / np.linalg.norm(self.rd_b, axis=1)[:, None]

        # express direction of vectors in body frame in {n}
        self.rd_n = (geom.Rzyx(*eta[3:6]).T.dot(self.rd_b.T)).T

        # Initialize intersection distance for each, if no intersect, take max_dist
        self._intersec_dist_fallback = np.full((self.n_rays,), self.max_dist)
        self.intersec_dist = copy.deepcopy(self._intersec_dist_fallback)

        # Get endpoint of all rays in {n} array(n, 3)
        self.end_pos_n = None
        # Do one update e.g. for the end position
        self.update(eta=eta)
        self.update_intersec()
        # Get number of reduced n_rays if it would be used
        self.blocksize_reduce = blocksize_reduce
        self.n_rays_reduced = self.intersec_dist_reduced.shape[0]


    def update(self, eta: np.ndarray) -> None:
        """
        Updates all the rays direction in {n} and the actual starting position,

        :param eta: array (6,) of position and attitude
        :return: None
        """
        pos = eta[0:3]
        attitude = eta[3:6]
        self.pos = pos
        self.rd_n = (geom.Rzyx(*attitude).dot(self.rd_b.T)).T
        # Normalize these vectors
        self.rd_n = self.rd_n / np.linalg.norm(self.rd_n, axis=1)[:, None]

    def update_intersec(self, intersec_dist: [None, np.ndarray] = None) -> None:
        """
        Update the intersection points and thus the end points of the radar, fallback is used in case None is parsed

        :param intersec_dist: array (n_rays,) with the intersection distances, negative and greater values than max
            dist are corrected automatically
        :return: None
        """
        # Save actual intersec distances, if it is needed somewhere elese
        if intersec_dist is None:
            intersec_dist = self._intersec_dist_fallback
        # Post edit result from intersec to make sure they are valid
        else:
            intersec_dist[(intersec_dist < 0) | (intersec_dist > self.max_dist)] = self.max_dist
        self.intersec_dist = intersec_dist
        # Update all end position points with intersection distance provided from outside
        self.end_pos_n = self.pos + self.rd_n * self.intersec_dist[:, None]

    @property
    def pos_arr(self):
        """
        Helper property in case needed array (n,3) for starting position

        :return: array (n,3), where the starting position is copied n times to create artificial array
        """
        return np.tile(self.pos, (self.n_rays, 1))

    @property
    def intersec_dist2d(self):
        return self.intersec_dist.reshape((self.n_vertical, self.n_horizontal))

    @property
    def intersec_dist_reduced(self):
        return block_reduce(self.intersec_dist2d, block_size=self.blocksize_reduce, func=np.max).flatten()

    def reset(self, eta: np.ndarray):
        """
        Helper function to reset radar
        """
        self.update(eta=eta)
        self.update_intersec()


class Ray:
    """
    Represents a Ray in 3d, can be initiated with a starting point, a vector for pointing its direction and maximum
    distance.

    DEPRECATED since other sensors are vectorized
    """

    def __init__(self, pos: np.ndarray, rd: np.ndarray, max_dis: float):
        self.pos = pos
        self.rd_b = rd / np.linalg.norm(rd)  # Make it a unit vector, assumed to be given in body frame
        self.rd_n = self.rd_b  # Unit vector in ned frame, needs to be calculated
        self.max_dis = max_dis
        self.intersec_dist = None  # Intersection distance, when available

    def get_point_from_dist(self, dist: float) -> np.ndarray:
        """
        Function to retrieve a point on the line with giving it a distance in {n}

        :param dist: distance from starting point
        :return: array (3,) for the point on the line
        """

        return self.pos + self.rd_n * dist
