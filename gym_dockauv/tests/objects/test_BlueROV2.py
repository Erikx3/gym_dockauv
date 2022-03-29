import unittest
import os

from objects.vehicles.BlueROV2 import BlueROV2


class TestBlueROV2(unittest.TestCase):
    """
    Setup that is always called before all other test functions to read in instance
    """
    def setUp(self):
        xml_path = os.path.join(os.path.dirname(__file__), 'test_BlueROV2.xml')
        self.BlueROV2 = BlueROV2(xml_path)


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
    This function is to test some state space matrices
    """
    def test_B_matrix_dimension(self):
        self.assertEqual(self.BlueROV2.B.shape[0], 6)
        self.assertGreaterEqual(self.BlueROV2.B.shape[1], 1)


if __name__ == '__main__':
    unittest.main()
