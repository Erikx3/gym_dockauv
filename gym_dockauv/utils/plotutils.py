import matplotlib.pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation

# Used for typehints
from typing import List
import numpy as np

from ..objects.shape import Shape
from .blitmanager import BlitManager

from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    """
    Publicly available class for 3d arrows

    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


# For seamless integration we add the arrow3D method to the Axes3D class.

# Add this function during runtime with: setattr(Axes3D, 'arrow3D', _arrow3D)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add a 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


# TODO: Think about adding more plots like input, state variables etc
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

        # Add a episode number (need to be added to the blit manager, so it is drawn correctly on each call)
        dummy = self.ax_path.annotate(
            f"{episode_nr}",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )

        self.bm.add_artists([self.ax_path.path_art, self.ax_path.head_art, dummy])

        # Static shapes (assumed for now, drawing these dynamically could slow down the system significantly):
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
        self.ax_path.path_art.set_data_3d(position[:, 0], position[:, 1], position[:, 2])

        # Update head dot
        self.ax_path.head_art.set_data_3d(position[-1, 0], position[-1, 1], position[-1, 2])

        # Update animation plot
        self.bm.update()

    def save_wrap_update_animation(self, step_nr: int, kwargs: dict) -> None:
        """
        Wrapper function to deal with saving animation

        :param step_nr: actual step number
        :param kwargs: kwargs, as they appear in save_animation()

        :return: None
        """

        for key, value in kwargs.items():
            if key == "position":
                self.update_path_animation(position=np.array(value[:step_nr + 1, :]))

    def save_animation(self, save_path: str, fps: int = 10, **kwargs) -> None:
        """
        Saves video as mp4 for the animation

        Depending on the amount of subplots etc, necessary arguments need to be given in kwargs.

        :param fps: fps of the saved video, if you want it in real time, enter, 1/dt here, where dt is the step size
        :param save_path: absolute path of where to store the video

        :Keyword Arguments:
            * *position* (``np.ndarray``) --
              array nx3, where n is the total number of all available position data points after the episode

        :return: None
        """
        ani = animation.FuncAnimation(
            self.fig, func=self.save_wrap_update_animation, fargs=(kwargs,))

        writer_video = animation.FFMpegWriter(fps=fps)

        # TODO: Make this a logger statement
        print(f"\nSave video at {save_path}")
        ani.save(save_path, writer=writer_video)
        # TODO: Stop showing Animation when saving!
