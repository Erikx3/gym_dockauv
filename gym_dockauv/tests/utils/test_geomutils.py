import numpy as np
import unittest

import gym_dockauv.utils.geomutils as geom


class TestGeomUtils(unittest.TestCase):

    def test_ssa(self):
        x = geom.ssa(np.array([3 * np.pi, 3 * np.pi - 0.001, np.pi/2, 0, -4/3 * np.pi, 10/3 * np.pi]))
        self.assertAlmostEqual(x[0], -np.pi)
        self.assertAlmostEqual(x[1], np.pi - 0.001)
        self.assertAlmostEqual(x[2], np.pi/2)
        self.assertAlmostEqual(x[3], 0)
        self.assertAlmostEqual(x[4], 2/3*np.pi)
        self.assertAlmostEqual(x[5], -2/3*np.pi)

    def test_Rzyx(self):
        test_vector_b = np.array([1, 0, 0])
        test_Theta = np.array([np.pi/4, np.pi/4, np.pi/4])
        test_R = geom.Rzyx(*test_Theta)
        test_vector_n = test_R.dot(test_vector_b)
        self.assertAlmostEqual(test_vector_n[0], 0.5)
        self.assertAlmostEqual(test_vector_n[1], 0.5)
        self.assertAlmostEqual(test_vector_n[2], -1/2**0.5)

    def test_Tzyx(self):
        test_vector_b1 = np.array([1, 0, 0])
        test_vector_b2 = np.array([0, 1, 0])
        test_Theta = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        test_T = geom.Tzyx(*test_Theta[:2])
        test_vector_n1 = test_T.dot(test_vector_b1)
        test_vector_n2 = test_T.dot(test_vector_b2)
        #print(test_vector_n1, test_vector_n2)
        self.assertAlmostEqual(test_vector_n1[0], 1.0)
        self.assertAlmostEqual(test_vector_n1[1], 0.0)
        self.assertAlmostEqual(test_vector_n1[2], 0.0)
        self.assertAlmostEqual(test_vector_n2[0], 1/2**0.5)
        self.assertAlmostEqual(test_vector_n2[1], 1/2**0.5)
        self.assertAlmostEqual(test_vector_n2[2], 1)


if __name__ == '__main__':
    unittest.main()
