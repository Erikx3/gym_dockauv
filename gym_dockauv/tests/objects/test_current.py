import unittest
import numpy as np
import os
import matplotlib.pyplot as plt

from gym_dockauv.objects.current import Current


class TestCurrent(unittest.TestCase):
    """
    Setup that is always called before all other test functions to read in instance
    """

    def setUp(self):
        self.current = Current(mu=0.01, V_min=0.5, V_max=1.0, Vc_init=0.5,
                               alpha_init=np.pi/4, beta_init=np.pi/4, white_noise_std=0.1, step_size=0.1)
        # Use consistent test values, interesting comb is mu=0.005 and white_noise_std=0.05


class TestCurrentFunc(TestCurrent):
    """
    Test functions after the initialization of the BlueROV2 (includes reading the xml file)
    """

    def test_get_current_NED(self):
        vel_current_NED = self.current.get_current_NED()
        #print(vel_current_NED)
        self.assertAlmostEqual(vel_current_NED[0], 1 / 4)
        self.assertAlmostEqual(vel_current_NED[1], 1 / (2*2 ** 0.5))
        self.assertAlmostEqual(vel_current_NED[2], 1 / 4)

    def test_sim(self):
        n_sim = 1000
        V_c_data = np.zeros(n_sim)
        for i in range(n_sim):
            V_c_data[i] = self.current.V_c
            self.current.sim()

        # Plot this function test and save it for inspection
        title = 'test_current.test_current.TestCurrentFunc.test_sim.png'
        plt.plot(np.arange(0, n_sim)*0.1, V_c_data, 'b.-', linewidth=0.5, markersize=1.0)
        plt.title(title)
        plt.xlabel("t [s]")
        plt.ylabel("V_c [m/s]")
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_result_files', title))
        print(f"\nSave plot at {save_path}")
        plt.savefig(save_path)
        plt.close()
        self.assertEqual(os.path.isfile(save_path), True)

