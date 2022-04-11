import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
import time

# Import standardized BluerOV2 class
from .objects.test_BlueROV2 import TestBlueROV2
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.utils.datastorage import EpisodeDataStorage


class TestIntegration(TestBlueROV2):
    """
    Testing some simple interaction and make integration of multiple submodules
    """

    def test_episode_simulation(self):
        """
        Integration test for all functionalities regarding a live simulation of the vehicle
        """

        # Just moving forward (standard initialization has a skew in pitch, that why it is -0.5 here
        action = np.array([1, 0, 0, -0.5, 0, 0])
        self.BlueROV2.step_size = 0.1
        # Reset nu_r here
        self.BlueROV2.state = np.zeros(12)
        # No current for here
        nu_c = np.zeros(6)
        # Number of simulation steps
        n_sim = 100
        episode_nr = 1234
        # Initialize animation
        epi_anim = EpisodeAnimation()
        ax = epi_anim.init_path_animation()
        epi_anim.add_episode_text(ax, episode_nr)
        # Some extra axes manipulation for testing
        title = "Integration_Test_Episode_Simulation"
        ax.set(title=title)

        # Initialize Data Storage
        path_fol = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_result_files'))
        epi_storage = EpisodeDataStorage(path_folder=path_fol, vehicle=self.BlueROV2, title=title, episode=episode_nr)

        # Simulate and update animation and storage
        for i in range(n_sim):
            # if i > 997:
            #     print("\nnu_r ", self.BlueROV2.state[6:])
            #     print("\nB ", self.BlueROV2.B.dot(self.BlueROV2.u))
            #     print("\nD ", self.BlueROV2.D(self.BlueROV2.state[6:]).dot(self.BlueROV2.state[6:]))
            #     print("\nC ", self.BlueROV2.C(self.BlueROV2.state[6:]).dot(self.BlueROV2.state[6:]))
            #     print("\nG ", self.BlueROV2.G(self.BlueROV2.state[:6]))
            self.BlueROV2.step(action, nu_c)
            epi_storage.update()

            epi_anim.update_path_animation(positions=epi_storage.positions, attitudes=epi_storage.attitudes)
            time.sleep(0.2)

        """Note on why the vehicle is slightly pitching in simulation: Even if we apply only force in x and z direction, 
        the Mass Matrix M_A contains off diagonal elements, since the Center of Origin is placed at the center of 
        Buoyancy. This means, we expect the vehicle to pitch, since we have a simplifid control matrix B that does 
        not account for this. Meaning, we apply a simple force in x at CO and not CG leads to a rotation about the y 
        axis """

        # Save this test file

        epi_storage.save()


if __name__ == '__main__':
    unittest.main()
