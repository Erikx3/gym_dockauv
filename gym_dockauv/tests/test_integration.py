import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
import time

# Import standardized BluerOV2 class
from gym_dockauv.tests.objects.test_BlueROV2 import TestBlueROV2
from gym_dockauv.utils.plotutils import EpisodeAnimation, EpisodeVisualization
from gym_dockauv.utils.datastorage import EpisodeDataStorage
from gym_dockauv.objects.shape import Capsule
from gym_dockauv.objects.current import Current

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
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Reset nu_r here
        self.BlueROV2.state = np.zeros(12)
        # Water current
        current = Current(mu=0.01, V_min=0.0, V_max=0.5, Vc_init=0.2,
                          alpha_init=0.0, beta_init=0.0, white_noise_std=0.1, step_size=self.BlueROV2.step_size)
        nu_c = current(self.BlueROV2.state)
        # nu_c = np.array([0, 0, 0, 0, 0, 0])
        # Number of simulation steps
        n_sim = 100
        episode_nr = 1234
        # Initialize animation
        epi_anim = EpisodeAnimation()
        ax = epi_anim.init_path_animation()
        epi_anim.add_episode_text(ax, episode_nr)
        # Add shape for testing
        cylinder = Capsule(position=np.array([0.5, 0.5, 0.5]), radius=0.15, vec_top=np.array([1, 1, 1]))
        epi_anim.add_shapes(ax, [cylinder])
        # Some extra axes manipulation for testing
        title = "Integration_Test_Episode_Simulation"
        ax.set(title=title)

        # Initialize Data Storage

        epi_storage = EpisodeDataStorage()
        epi_storage.set_up_episode_storage(path_folder=PATH_FOL, vehicle=self.BlueROV2,
                                           step_size=self.BlueROV2.step_size, nu_c_init=nu_c, shapes=[cylinder],
                                           title=title, episode=episode_nr)
        epi_storage.file_save_name = os.path.join(PATH_FOL, f"{STORAGE_NAME}.pkl")

        # Simulate and update animation and storage
        for i in range(n_sim):

            # eta = self.BlueROV2.state[:6]
            # nu_r = self.BlueROV2.state[6:]
            # nu_r_dot = self.BlueROV2.M_inv.dot(
            #     self.BlueROV2.B(nu_r).dot(self.BlueROV2.u)
            #     - self.BlueROV2.D(nu_r).dot(nu_r)
            #     - self.BlueROV2.C(nu_r).dot(nu_r)
            #     - self.BlueROV2.G(eta))
            # print("B: \n", self.BlueROV2.B(nu_r).dot(self.BlueROV2.u))
            # print("D: \n", -self.BlueROV2.D(nu_r).dot(nu_r))
            # print("C: \n", -self.BlueROV2.C(nu_r).dot(nu_r))
            # print("G: \n", -self.BlueROV2.G(eta))
            # print("nu_r_dot: \n", nu_r_dot)
            # print("nu_r: \n", nu_r)

            # Simulate current
            current.sim()
            nu_c = current(self.BlueROV2.state)
            # Simulate
            self.BlueROV2.step(action, nu_c)
            # Update data storage
            epi_storage.update(nu_c=nu_c)
            # Update animation
            epi_anim.update_path_animation(positions=epi_storage.positions, attitudes=epi_storage.attitudes)
            # time.sleep(1)


        """Note on why the vehicle is slightly pitching in simulation: Even if we apply only force in x and z direction,
        the Mass Matrix M_A contains off diagonal elements, since the Center of Origin is placed at the center of
        Buoyancy. This means, we expect the vehicle to pitch, since we have a simplified control matrix B that does
        not account for this. Meaning, we apply a simple force in x at CO and not CG leads to a rotation about the y
        axis 
        
        Note 2: Dynamic coupling of Restoring forces, CO at CB, Coriolis Forces and Damping forces lead to swing
        """

        # Save this test file
        print(f"Save pickle file for episode data storge at {epi_storage.file_save_name}")
        epi_storage.save()
        del epi_anim

    def test_post_flight_visualization(self):
        epi_stor = EpisodeDataStorage()
        epi_stor.load(os.path.join(PATH_FOL, f"{STORAGE_NAME}.pkl"))
        epi_stor.plot_epsiode_states_and_u()
        plt.savefig(os.path.join(PATH_FOL, f"{STORAGE_NAME}_Plot.png"))
        plt.close('all')
        epi_stor.plot_episode_animation(t_per_step=None, title="Test Post Flight Visualization")
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
