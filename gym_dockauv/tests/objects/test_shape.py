import numpy as np
import unittest
from gym_dockauv.objects import shape


class TestShape(unittest.TestCase):

    def setUp(self) -> None:
        self.point = np.array([0.5, 0.5, 0.5])
        self.l11 = np.array([1, 1, 1])
        self.l12 = np.array([1, 1, 0])

        self.point2 = np.array([-1, -1, -2.5])
        self.l21 = np.array([0.0, 0.0, 0.0])
        self.l22 = np.array([2.0, 2.0, 0.0])


class TestShapeFunc(TestShape):

    def test_dist_line_point(self):
        self.assertAlmostEqual(0.5**0.5, shape.dist_line_point(self.point, self.l11, self.l12))
        self.assertAlmostEqual(8.25**0.5, shape.dist_line_point(self.point2, self.l21, self.l22))

    def test_collision_capsule_sphere(self):
        cap_rad = 1
        sph_rad = 0.5
        self.assertEqual(True, shape.collision_capsule_sphere(self.l11, self.l12, cap_rad, self.point, sph_rad))
        self.assertEqual(False, shape.collision_capsule_sphere(self.l21, self.l22, cap_rad, self.point2, sph_rad))

    def test_collision_sphere_spheres(self):
        pos1 = np.array([0, 0, 0])
        rad1 = 1
        pos2 = np.array([[3, 0, 0], [1, 1, 1]])
        rad2 = np.array([1, 1])
        self.assertEqual(True, shape.collision_sphere_spheres(pos1, rad1, pos2, rad2))
        rad3 = np.array([1, 0.5])
        self.assertEqual(False, shape.collision_sphere_spheres(pos1, rad1, pos2, rad3))

    def test_intersec_dist_line_capsule(self):
        cap_rad = 1
        dist = shape.intersec_dist_line_capsule(l1=self.l21, ld=(self.l22-self.l21), cap1=self.l11, cap2=self.l12,
                                                cap_rad=cap_rad)
        self.assertAlmostEqual(dist, 2**0.5-1)

        # Test for expected behavior, when line points in on direction and capsule is behind, see comment on function
        dist2 = shape.intersec_dist_line_capsule(l1=self.l21, ld=np.array([-2.0, -2.0, 0.0]), cap1=self.l11,
                                                 cap2=self.l12, cap_rad=cap_rad)
        self.assertAlmostEqual(dist2, -(2**0.5+1))

        # Test for no intersection at all, should return -np.inf
        dist3 = shape.intersec_dist_line_capsule(l1=self.l21, ld=np.array([-2.0, 2.0, 0.0]), cap1=self.l11,
                                                 cap2=self.l12, cap_rad=cap_rad)
        self.assertAlmostEqual(dist3, -np.inf)

    def test_intersec_dist_line_capsule_vectorized(self):
        cap_rad = 1
        l1 = np.vstack([self.l21, self.l21])
        ld = np.vstack([(self.l22-self.l21), np.array([-2.0, -2.0, 0.0])])
        dist = shape.intersec_dist_line_capsule_vectorized(l1=l1, ld=ld, cap1=self.l11, cap2=self.l12, cap_rad=cap_rad)
        self.assertAlmostEqual(dist[0], 2**0.5-1)
        self.assertAlmostEqual(dist[1], -(2**0.5+1))

    def test_intersec_dist_lines_spheres_vectorized(self):
        l1 = np.array([
            [0, 0, 3],
            [0, -2, 0],
            [2, 2, 0],
            [-5, 0, 0]
        ])
        ld = np.array([
            [0, 0, -2],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        center = np.array([
            [0, 0, 0],
            [-2, 0, 0]
        ])
        rad = np.array([1, 0.5])
        dist = shape.intersec_dist_lines_spheres_vectorized(l1=l1, ld=ld, center=center, rad=rad)
        self.assertAlmostEqual(dist[0], 2.0)
        self.assertAlmostEqual(dist[1], 1.0)
        self.assertAlmostEqual(dist[2], -np.inf)
        self.assertAlmostEqual(dist[3], 2.5)

    def test_vec_line_point(self):
        # Some 2d tests first
        po_2d = np.array([0, 0])
        l1_2d = np.array([-2, 1])
        l2_2d = np.array([2, 1])
        res_2d = shape.vec_line_point(po_2d, l1_2d, l2_2d)
        self.assertAlmostEqual(res_2d[0], 0.0)
        self.assertAlmostEqual(res_2d[1], 1.0)

        # Some 3d test
        po_3d = np.array([0, 0, 1])
        l1_3d = np.array([-2, 1, 2])
        l2_3d = np.array([2, 1, 0])
        res_3d = shape.vec_line_point(po_3d, l1_3d, l2_3d)
        self.assertAlmostEqual(res_3d[0], 0.0)
        self.assertAlmostEqual(res_3d[1], 1.0)
        self.assertAlmostEqual(res_3d[2], 0.0)


if __name__ == '__main__':
    unittest.main()
