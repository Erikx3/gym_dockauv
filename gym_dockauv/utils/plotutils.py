import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Used for typehints
from typing import List

import numpy as np

from ..objects.shape import Shape

from .blitmanager import BlitManager


# TODO: Think about adding more plots like input, state etc.
class EpisodeAnimation:

    def __init__(self):
        self.fig = plt.figure()
        self.bm = BlitManager(self.fig.canvas, [])

        # Some predefinitions
        self.ax_path = None

    def __del__(self):
        plt.close(self.fig)

    # TODO: Add attitude visualization
    def init_path_animation(self, shapes: List[Shape], episode_nr: int) -> matplotlib.pyplot.axes():
        """
        Initialization of the live animation plot for the path, includes:
        - shapes (assumed to be static)
        - path + head

        :return: axes
        """
        # Create Axis
        self.ax_path = self.fig.add_subplot(projection="3d")

        # Create lines initially without data
        self.ax_path.path_art = self.ax_path.plot([], [], [], 'g--', alpha=1.0, animated=True)[0]

        # Create dot instance
        self.ax_path.head_art = self.ax_path.plot([], [], [], 'b.', alpha=1.0, markersize=10, animated=True)[0]

        self.bm.add_artists([self.ax_path.path_art, self.ax_path.head_art])

        # TODO: This alwys disappears for the live animation, fix it!
        # Static elements:
        # add a episode number
        self.ax_path.annotate(
            f"{episode_nr}",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )

        for shape in shapes:
            self.ax_path.plot_surface(*shape.get_plot_variables(), color='r', alpha=1.00)

        # Pause for initialization to be done
        plt.pause(0.5)

        return self.ax_path

    def update_path_animation(self, position: np.ndarray) -> None:
        """
        Update the path animation plot by updating the according elements

        :param position: array nx3, where n is the number of all available position data points so far in this episode
        :return: None
        """
        # Update path line
        self.ax_path.path_art.set_data(position[:, :2].T)
        self.ax_path.path_art.set_3d_properties(position[:, 2])

        # Update head dot
        # Note: Little workaround, since set_data only except 2d np arrays
        self.ax_path.head_art.set_data(np.array(position[-1, :2].T)[:, None])
        self.ax_path.head_art.set_3d_properties(np.array(position[-1, 2]))  # Note: Always 1d array fine here

        # Update animation plot
        self.bm.update()

    def save_wrap_update_path_animation(self, step_nr: int, position: np.ndarray) -> None:
        """
        Wrapper function to deal with saving animation

        :param step_nr: actual step number
        :param position: array nx3, where n is the total number of all available position data points after the episode

        :return: None
        """
        self.update_path_animation(position=np.array(position[:step_nr+1, :]))

    def save_path_animation(self, position: np.ndarray, save_path: str, interval: int = 100) -> None:
        """
        Saves video as mp4 for the animation

        :param position: array nx3, where n is the total number of all available position data points after the episode
        :param interval: of each time step in ms for the video (can be set to dt, OR smaller if video would be too long)
        :param save_path: absolute path of where to store the video
        :return: None
        """

        ani = animation.FuncAnimation(
            self.fig, func=self.save_wrap_update_path_animation,
            frames=position.shape[0], fargs=(position,), interval=interval)

        writer_video = animation.FFMpegWriter(fps=30)

        print(f"\nSave video at {save_path}")
        ani.save(save_path, writer=writer_video)
        # TODO Rework saving (fps, interval etc)
        # TODO: Think about the structure (since save_animation would mean for whole figure maybe, save with **kwargs!)
