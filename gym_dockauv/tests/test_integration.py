import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
import time

# Import standardized BluerOV2 class
from gym_dockauv.tests.objects.test_BlueROV2 import TestBlueROV2
from gym_dockauv.utils.plotutils import EpisodeAnimation, EpisodeVisualization
from gym_dockauv.utils.datastorage import EpisodeDataStorage
from gym_dockauv.objects.shape import Capsule, intersec_dist_line_capsule_vectorized, Sphere, Spheres, intersec_dist_lines_spheres_vectorized
from gym_dockauv.objects.current import Current
from gym_dockauv.objects.sensor import Radar

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
        action = np.array([0.00, -0.0, -0.0, 0.0, -0.0, 0.0])
        # Reset nu_r here
        self.BlueROV2.state = np.zeros(12)

        # Water current
        current = Current(mu=0.01, V_min=0.0, V_max=0.5, Vc_init=0.1,
                          alpha_init=0.0, beta_init=0.0, white_noise_std=0.1, step_size=self.BlueROV2.step_size)
        nu_c = current(self.BlueROV2.attitude)
        #nu_c = np.array([0, 0, 0, 0, 0, 0])

        # Add sensor suite for testing
        eta = np.array([0, 0, 0, 0, 0, 0])
        freq = 1
        alpha = 30 * np.pi / 180
        beta = 20 * np.pi / 180
        ray_per_deg = 5 * np.pi / 180
        radar = Radar(eta=eta, freq=freq, alpha=alpha,
                      beta=beta, ray_per_deg=ray_per_deg, max_dist=2)

        # Number of simulation steps
        n_sim = 200
        episode_nr = 1234

        # Initialize animation
        epi_anim = EpisodeAnimation()
        ax = epi_anim.init_path_animation()
        epi_anim.add_episode_text(ax, episode_nr)
        # Add shape for testing
        capsule = Capsule(position=np.array([1.5, 0.0, -0.0]), radius=0.25, vec_top=np.array([1.5, 0.0, -0.8]))
        sphere1 = Sphere(position=np.array([1.8, 0.0, -1.2]), radius=0.3)
        sphere2 = Sphere(position=np.array([2.3, 0.0, -1.5]), radius=0.3)
        spheres = Spheres(spheres=[sphere1, sphere2])
        epi_anim.add_shapes(ax, [capsule, sphere1, sphere2])
        # Some extra axes manipulation for testing
        title = "Integration_Test_Episode_Simulation"
        ax.set(title=title)
        # Initialize sensor animation
        epi_anim.init_radar_animation(radar.n_rays)

        # Initialize Data Storage
        epi_storage = EpisodeDataStorage()
        epi_storage.set_up_episode_storage(path_folder=PATH_FOL, vehicle=self.BlueROV2,
                                           step_size=self.BlueROV2.step_size, nu_c_init=nu_c,
                                           shapes=[capsule, sphere1, sphere2],
                                           radar=radar, title=title, episode=episode_nr)
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
            nu_c = current(self.BlueROV2.attitude)
            # Simulate vehicle
            self.BlueROV2.step(action, nu_c)
            # Update sensors
            radar.update(self.BlueROV2.eta)  # Update radar attitude first
            # Then calculate intersection with capsule
            i_dist_cap = intersec_dist_line_capsule_vectorized(
                l1=radar.pos_arr, ld=radar.rd_n, cap1=capsule.vec_bot, cap2=capsule.vec_top,
                cap_rad=capsule.radius)
            # Then calculate intersection with spheres
            i_dist_sph = intersec_dist_lines_spheres_vectorized(
                l1=radar.pos_arr, ld=radar.rd_n, center=spheres.position, rad=spheres.radius)
            # Get the smaller positive value
            i_dist = np.vstack([i_dist_cap, i_dist_sph]).T
            i_dist = i_dist[np.arange(i_dist.shape[0]), np.where(i_dist > 0, i_dist, np.inf).argmin(axis=1)]
            radar.update_intersec(intersec_dist=i_dist)  # Update radar intersections
            # Update data storage
            epi_storage.update(nu_c=nu_c)
            # Update animation of vehicle
            epi_anim.update_path_animation(positions=epi_storage.positions, attitudes=epi_storage.attitudes)
            # Update animation of sensors
            epi_anim.update_radar_animation(radar.pos, radar.end_pos_n)
            #time.sleep(0.1)

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

        # Uncomment if you want to save a video
        # temp_title = "temp_title"
        # ax.set(title=f"{temp_title}")
        # save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_result_files', temp_title+'.mp4'))
        # epi_anim.save_animation(save_path=save_path, fps=int(1/self.BlueROV2.step_size),
        #                         frames=epi_storage.positions.shape[0],
        #                         positions=epi_storage.positions, attitudes=epi_storage.attitudes,
        #                         end_pos=epi_storage.storage["radar"])
        del epi_anim

    def test_post_flight_visualization(self):
        # SO far without sensors, since it could be reanimated if needed, but not necessary for now
        epi_stor = EpisodeDataStorage()
        epi_stor.load(os.path.join(PATH_FOL, f"{STORAGE_NAME}.pkl"))
        epi_stor.plot_epsiode_states()
        plt.savefig(os.path.join(PATH_FOL, f"{STORAGE_NAME}_Plot.png"))
        plt.close('all')
        epi_stor.plot_episode_animation(t_per_step=None, title="Test Post Flight Visualization")
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
