import unittest
import os
import numpy as np

from gym_dockauv.objects.vehicles.BlueROV2 import BlueROV2


class TestBlueROV2(unittest.TestCase):
    """
    Setup that is always called before all other test functions to read in instance
    """
    def setUp(self):
        xml_path = os.path.join(os.path.dirname(__file__), 'test_BlueROV2.xml')
        self.BlueROV2 = BlueROV2(xml_path)
        self.nu_r = np.array([3, 2, 1, 0.3, 0.2, 0.1])


class TestInit(TestBlueROV2):
    """
    Test functions after the initialization of the BlueROV2 (includes reading the xml file)
    """
    # Test, if values from xml are actually read
    def test_initial_mass(self):
        self.assertEqual(self.BlueROV2.m, 11.5)

    def test_initial_buoyancy(self):
        self.assertEqual(self.BlueROV2.BY, 114.8)

    def test_initial_name(self):
        self.assertEqual(self.BlueROV2.name, "BlueROV2")

    def test_initial_X_udot(self):
        self.assertEqual(self.BlueROV2.X_udot, -5.5)

    def test_initial_Y_vv(self):
        self.assertEqual(self.BlueROV2.Y_vv, -21.66)

    # Test, if values are initialized from parent class as zero and not changed
    def test_initial_x_G(self):
        self.assertEqual(self.BlueROV2.x_G, 0.0)


class TestStateSpace(TestBlueROV2):
    """
    Test functions to test some state space matrices
    """
    def test_B_matrix_dimension(self):
        self.assertEqual(self.BlueROV2.B.shape[0], 6)
        self.assertGreaterEqual(self.BlueROV2.B.shape[1], 1)

    def test_C_A_matrix(self):
        # Some hand calculated checks first:
        self.assertAlmostEqual(self.BlueROV2.C_A(self.nu_r)[0, 4], 14.57)
        self.assertAlmostEqual(self.BlueROV2.C_A(self.nu_r)[2, 3], 25.4)
        self.assertAlmostEqual(self.BlueROV2.C_A(self.nu_r)[5, 4], -0.036)

        # We take the hand calculated solution completely and then compare it with our solution
        u = self.nu_r[0]
        v = self.nu_r[1]
        w = self.nu_r[2]
        p = self.nu_r[3]
        q = self.nu_r[4]
        r = self.nu_r[5]

        C_11 = np.zeros((3, 3))
        C_12 = np.array([[0, -self.BlueROV2.Z_wdot * w, self.BlueROV2.Y_vdot * v],
                         [self.BlueROV2.Z_wdot * w, 0, -self.BlueROV2.X_udot * u],
                         [-self.BlueROV2.Y_vdot * v, self.BlueROV2.X_udot * u, 0]])

        C_21 = C_12.copy()
        C_22 = np.array([[0, -self.BlueROV2.N_rdot * r, self.BlueROV2.M_qdot * q],
                         [self.BlueROV2.N_rdot * r, 0, -self.BlueROV2.K_pdot * p],
                         [-self.BlueROV2.M_qdot * q, self.BlueROV2.K_pdot * p, 0]])

        C_A = np.vstack([np.hstack([C_11, C_12]), np.hstack([C_21, C_22])])

        # print("\n", self.BlueROV2.C_A(self.nu_r), "\n", C_A)
        self.assertIsNone(np.testing.assert_array_equal(self.BlueROV2.C_A(self.nu_r), C_A))

    def test_I_b_matrix(self):
        # print("\n", self.BlueROV2.I_b)
        self.assertAlmostEqual(self.BlueROV2.I_b[0, 0], 0.2146)
        self.assertAlmostEqual(self.BlueROV2.I_b[1, 1], 0.2496)
        self.assertAlmostEqual(self.BlueROV2.I_b[2, 2], 0.245)

    def test_C_RB_matrix(self):
        # Some hand calculated checks (important: we use the linear independent form):
        # print("\n", self.BlueROV2.C_RB(self.nu_r), "\n", self.BlueROV2.M_RB)
        self.assertAlmostEqual(self.BlueROV2.C_RB(self.nu_r)[0, 3], 0.023)
        self.assertAlmostEqual(self.BlueROV2.C_RB(self.nu_r)[2, 3], -0.069)
        self.assertAlmostEqual(self.BlueROV2.C_RB(self.nu_r)[5, 4], -0.06438)

        # print("\n", self.BlueROV2.G(np.array([1, 1, 1, 0, 0, 0])), "\n", self.BlueROV2.D(self.nu_r))


if __name__ == '__main__':
    unittest.main()
