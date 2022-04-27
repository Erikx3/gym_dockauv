import unittest
import os
import numpy as np
from scipy.integrate import solve_ivp

from gym_dockauv.objects.vehicles.BlueROV2 import BlueROV2


class TestBlueROV2(unittest.TestCase):
    """
    Setup that is always called before all other test functions to read in instance
    """

    def setUp(self):
        xml_path = os.path.join(os.path.dirname(__file__), 'test_BlueROV2.xml')
        self.BlueROV2 = BlueROV2(xml_path)
        self.BlueROV2.step_size = 0.05
        # Use consistent test values
        self.nu_r = np.array([3, 2, 1, 0.3, 0.2, 0.1])
        self.BlueROV2.set_B(np.identity(6))
        # self.BlueROV2.set_B(np.array([
        #     [1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0]
        # ]))
        self.BlueROV2.set_u_bound(np.array([
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-1, 3],
            [-1, 1],
            [-1, 1]]))


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
        # No nu needed for the BlueROv
        self.assertEqual(self.BlueROV2.B(None).shape[0], 6)
        self.assertGreaterEqual(self.BlueROV2.B(None).shape[1], 1)

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

    def test_G_matrix(self):
        test_eta = np.array([0, 0, 0, 0, 0, 0])
        test_eta_moved = np.array([3, 2, 1, 0.3, 0.2, 0.1])

        # Check if all values except for the Z force is zero
        self.assertEqual(self.BlueROV2.G(test_eta)[0], 0)
        self.assertEqual(self.BlueROV2.G(test_eta)[1], 0)
        self.assertNotEqual(self.BlueROV2.G(test_eta)[2], 0)

        # Check by movement all are non-zero except last element
        self.assertNotEqual(self.BlueROV2.G(test_eta_moved)[3], 0)
        self.assertNotEqual(self.BlueROV2.G(test_eta_moved)[4], 0)
        self.assertEqual(self.BlueROV2.G(test_eta_moved)[5], 0)

        # print("\n", self.BlueROV2.G(test_eta), "\n", self.BlueROV2.G(test_eta_moved))
        # print("\n", self.BlueROV2.D(self.nu_r))


class TestAUVSim(TestBlueROV2):
    """
    Test functions for the AUV Sim simulation functionalities (unit tests, integration tests are in another file)
    """

    def test_unnormalize_input(self):
        input_test = np.array([-1.0, -0.5, 0.0, 0.5, 0.5, 1.0])
        # print("\n", self.BlueROV2.unnormalize_input(input_test))
        # Will be un-normalized with the u_bound, which is picked non-symmetrical for test purposes.
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[0], -5)
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[1], -2.5)
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[2], 0.0)
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[3], 2.0)
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[4], 0.5)
        self.assertEqual(self.BlueROV2.unnormalize_input(input_test)[5], 1.0)

    def test_sim_ode(self):
        """
        Comparison of ODE Solver solutions for the BlueROV2 simulation
        """
        # Just moving forward
        action = np.array([1, 0, 0, -0.5, 0, 0])
        # Pick smaller stepsize
        self.BlueROV2.step_size = 0.01
        # Reset nu_r here
        self.BlueROV2.state = np.zeros(12)
        # No current for now
        nu_c = np.zeros(6)
        # Number of simulation
        n_sim = 100

        # Make sure starting position is as we expect
        # print(self.BlueROV2.position)

        # Simulate own implementation and save results
        for _ in range(n_sim):
            self.BlueROV2.step(action, nu_c)
        state = self.BlueROV2.state
        # print("\n Position: ", self.BlueROV2.position)
        # print("\n u: ", self.BlueROV2.u)

        # Now compare with python ode solution, reset first
        self.BlueROV2.u = np.zeros(6)
        self.BlueROV2.state = np.zeros(12)
        for _ in range(n_sim):
            self.BlueROV2.u = self.BlueROV2.lowpassfilter.apply_lowpass(self.BlueROV2.unnormalize_input(action),
                                                                        self.BlueROV2.u)
            res = solve_ivp(fun=self.BlueROV2.state_dot, t_span=[0, self.BlueROV2.step_size],
                            y0=self.BlueROV2.state, t_eval=[self.BlueROV2.step_size], method='RK45', args=(nu_c,))
            self.BlueROV2.state = res.y.flatten()
        # print("\n Position: ", self.BlueROV2.position)
        # print("\n u: ", self.BlueROV2.u)

        # Finally, compare results
        self.assertIsNone(np.testing.assert_array_almost_equal(self.BlueROV2.state, state))


if __name__ == '__main__':
    unittest.main()
