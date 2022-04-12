import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
import time

# Import standardized BluerOV2 class
from gym_dockauv.tests.objects.test_BlueROV2 import TestBlueROV2
from gym_dockauv.utils.plotutils import EpisodeAnimation, EpisodeVisualization
from gym_dockauv.utils.datastorage import EpisodeDataStorage

# Only here: Overwrite storage file name after init for making consitent tests.
STORAGE_NAME = "YYYY_MM_DDTHH_MM_SS__episodeX__Integration_Test"
PATH_FOL = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_result_files'))


class TestIntegration(TestBlueROV2):
    """
    Testing some simple interaction and make integration of multiple submodules
    """

    def test_episode_simulation(self):
        """
        Integration test for all functionalities regarding a live simulation of the vehicle
        """

        # Just moving forward (standard initialization has a skew in pitch, that why it is -0.5 here)
        action = np.array([1, 0, 0, -0.5, 0, 0])
        self.BlueROV2.step_size = 0.1
        # Reset nu_r here
        self.BlueROV2.state = np.zeros(12)
        # No current for now
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

        epi_storage = EpisodeDataStorage()
        epi_storage.set_up_episode_storage(path_folder=PATH_FOL, vehicle=self.BlueROV2,
                                           step_size=self.BlueROV2.step_size, title=title, episode=episode_nr)
        epi_storage.file_save_name = os.path.join(PATH_FOL, f"{STORAGE_NAME}.pkl")

        # Simulate and update animation and storage
        for i in range(n_sim):
            # Simulate
            self.BlueROV2.step(action, nu_c)
            # Update data storage
            epi_storage.update()
            # Update animation
            epi_anim.update_path_animation(positions=epi_storage.positions, attitudes=epi_storage.attitudes)
            #time.sleep(0.02)

        """Note on why the vehicle is slightly pitching in simulation: Even if we apply only force in x and z direction,
        the Mass Matrix M_A contains off diagonal elements, since the Center of Origin is placed at the center of
        Buoyancy. This means, we expect the vehicle to pitch, since we have a simplified control matrix B that does
        not account for this. Meaning, we apply a simple force in x at CO and not CG leads to a rotation about the y
        axis """

        # Save this test file
        print(f"Save pickle file for episode data storge at {epi_storage.file_save_name}")
        epi_storage.save()
        del epi_anim

    def test_post_flight_visualization(self):
        epi_vis = EpisodeVisualization(os.path.join(PATH_FOL, f"{STORAGE_NAME}.pkl"))
        epi_vis.plot_episode_states_and_u()
        plt.savefig(os.path.join(PATH_FOL, f"{STORAGE_NAME}_Plot.png"))
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
