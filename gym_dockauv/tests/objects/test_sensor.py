import numpy as np
import unittest
from gym_dockauv.objects import sensor


class TestSensor(unittest.TestCase):

    def setUp(self) -> None:
        self.eta = np.array([0, 0, 0, 0, 0, 0])
        self.freq = 1
        self.alpha = 30*np.pi/180
        self.beta = 20*np.pi/180
        self.ray_per_deg = 5*np.pi/180
        self.radar = sensor.Radar(eta=self.eta, freq=self.freq, alpha=self.alpha,
                                  beta=self.beta, ray_per_deg=self.ray_per_deg, max_dist=5)


class TestSensorFunc(TestSensor):
    def test_init(self):
        n_rays = self.alpha//self.ray_per_deg * self.beta//self.ray_per_deg
        self.assertEqual(self.radar.alpha.shape[0], n_rays)
        self.assertEqual(self.radar.alpha.beta[0], n_rays)
        self.assertEqual(self.radar.rd_n.shape[0], n_rays)
        self.assertEqual(self.radar.rd_n.shape[1], n_rays)

        # TODO: Check more from Initialization of sensors (only done by hand so far)


if __name__ == '__main__':
    unittest.main()
