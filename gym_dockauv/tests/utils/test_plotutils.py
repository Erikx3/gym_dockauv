import time

import matplotlib.pyplot as plt
import numpy as np
import unittest
import os
from gym_dockauv.objects.shape import Sphere, Capsule
from gym_dockauv.utils.plotutils import EpisodeAnimation


class TestPlotUtils(unittest.TestCase):

    def test_episode_animation(self):
        cylinder = Capsule(position=np.array([0.5, 0.5, 0.5]), radius=0.15, vec_top=np.array([1.0, 1.0, 1.0]))
        epi_anim = EpisodeAnimation()
        ax = epi_anim.init_path_animation()
        epi_anim.add_shapes(ax, [cylinder])
        epi_anim.add_episode_text(ax, 123)

        # Some extra axes manipulation for testing
        title = "Testing_Episode_Animations"
        ax.set(title=title)

        # Fake episodic process and test update function
        num_steps = 100
        positions = random_walk(num_steps, max_step=0.02)
        attitudes = random_walk(num_steps, max_step=0.05)
        for i in range(num_steps):
            # Fake available data so far
            position = positions[:i + 1, :]
            attitude = attitudes[:i + 1, :]
            epi_anim.update_path_animation(positions=position, attitudes=attitude)
            # Some pausing for seeing the plot:
            # plt.pause(0.01)

        # Test saving the animation right after
        ax.set(title=f"{title}_Save_Video")
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_result_files', title+'.mp4'))
        epi_anim.save_animation(save_path=save_path, fps=20, frames=positions.shape[0],
                                positions=positions, attitudes=attitudes)

        plt.close(epi_anim.fig)

    def test_plot_shape(self):
        title = 'test_plotutils.TestPlotUtils.test_plot_shape.png'
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set(title=title)
        ax.set_proj_type('ortho')

        # Sphere
        sphere = Sphere(position=np.array([0.2, 0.2, 0.7]), radius=.4)
        # Create Sphere instance Note: Hard to update Sphere position only for Blitmanager, rather need to update
        # whole axis (set_verts did not work out, there is a lot under the hood of plot_surface)
        surface_sphere = ax.plot_surface(*sphere.get_plot_variables()[0], color='b', alpha=0.7)

        # Capsule
        capsule = Capsule(position=np.array([1, 1, 0]), radius=0.25, vec_top=np.array([0.5, 0.5, 0.5]))
        surface_cap = ax.plot_surface(*capsule.get_plot_variables()[0], color='r', alpha=1)
        surface_cap_sph1 = ax.plot_surface(*capsule.get_plot_variables()[1], color='r', alpha=0.5)
        surface_cap_sph2 = ax.plot_surface(*capsule.get_plot_variables()[2], color='r', alpha=0.5)

        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_result_files', title))
        print(f"\nSave plot at {save_path}")
        plt.savefig(save_path)
        plt.close(fig)
        self.assertEqual(os.path.isfile(save_path), True)


# Function used to simulate a random walk
def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array (Used here for plotting simple things)."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


# Below here: deprecated first simple tests

# def update(num, walk, line, fr_number, dot):
#     # NOTE: there is no .set_data() for 3 dim data...
#     # Update line
#     line.set_data(walk[:num, :2].T)
#     line.set_3d_properties(walk[:num, 2])
#     # Update text
#     fr_number.set_text("frame: {j}".format(j=num))
#
#     # Update dot
#     dot.set_data(np.array(walk[num, :2].T)[:, None])
#     dot.set_3d_properties(np.array(walk[num, 2]))
#     return line, dot


# def create_example_artists(num_steps, ax):
#     # Add walk data
#     walk = random_walk(num_steps)
#
#     # Create lines initially without data
#     line = ax.plot([], [], [], 'r--', alpha=0.8, animated=True)[0]
#
#     # add a frame number
#     fr_number = ax.annotate(
#         "0",
#         (0, 1),
#         xycoords="axes fraction",
#         xytext=(10, -10),
#         textcoords="offset points",
#         ha="left",
#         va="top",
#         animated=True,
#     )
#
#     # Create dot instance
#     dot = ax.plot([], [], [], 'b.', alpha=0.8, markersize=10, animated=True)[0]
#
#     return walk, line, fr_number, dot


# class Deprecated:
#
#     @staticmethod
#     def test_blitmanager():
#         """
#         Test function for the blit manager with minimum example to see, whether it works on this text computer.
#         """
#
#         # Example use with example adopted from:
#         # https: // matplotlib.org / stable / gallery / animation / random_walk.html
#
#         np.random.seed(3)
#         # Attaching 3D axis to the figure
#         fig = plt.figure()
#         ax = fig.add_subplot(projection="3d")
#
#         # Data: 100 random walks as (num_steps, 3) arrays
#         num_steps = 100
#         # Create example artists
#         walk, line, fr_number, dot = create_example_artists(num_steps=num_steps, ax=ax)
#
#         # Setting the axes properties
#         ax.set(xlim3d=(0, 1), xlabel='X')
#         ax.set(ylim3d=(0, 1), ylabel='Y')
#         ax.set(zlim3d=(0, 1), zlabel='Z')
#         ax.set(title="Testing Blit Manager and Animations")
#         ax.set_proj_type('ortho')
#
#         bm = BlitManager(fig.canvas, [line, fr_number, dot])
#         # make sure our window is on the screen and drawn
#         plt.show(block=False)
#         plt.pause(0.5)
#
#         for j in range(num_steps):
#             # update the artists
#             update(j, walk, line, fr_number, dot)
#
#             # Setting the axes properties dynamically, need to add whole ax to BlitManager
#             # If needed, would reduce to a minimum level of usage to not slow down visualization
#             # ax.set(xlim3d=(0-j/num_steps, 1+j/num_steps), xlabel='X')
#             # ax.set(ylim3d=(0-j/num_steps, 1+j/num_steps), ylabel='Y')
#             # ax.set(zlim3d=(0-j/num_steps, 1+j/num_steps), zlabel='Z')
#
#             # Let the blitting manager update
#             bm.update()
#             # Optional for slowing down speed
#             plt.pause(0.05)
#
#         plt.close(fig)
#         # If made it until this point, give some kind of unit test feedback
#         # self.assertEqual(1, 1)
#
#     @staticmethod
#     def test_save_animation(self):
#         """
#         Testing whether animation can be saved as a video
#         """
#         # here add example of how to save video in a clean way after "live" animating stuff (With the saved data!)
#         title = 'test_plotutils.TestPlotUtils.test_save_animation.mp4'
#         fig = plt.figure()
#         ax = fig.add_subplot(projection="3d")
#         ax.set(title=title)
#         fps = 10
#         # Data: 100 random walks as (num_steps, 3) arrays
#         num_steps = 100
#         # Create example artists
#         walk, line, fr_number, dot = create_example_artists(num_steps=num_steps, ax=ax)
#
#         ani = animation.FuncAnimation(
#             fig, func=update, frames=num_steps, fargs=(walk, line, fr_number, dot), interval=100)
#
#         writer_video = animation.FFMpegWriter(fps=fps)
#         save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_result_files', title))
#         print(f"\nSave video at {save_path}")
#         ani.save(save_path, writer=writer_video)
#         #self.assertEqual(os.path.isfile(save_path), True)

if __name__ == '__main__':
    unittest.main()
