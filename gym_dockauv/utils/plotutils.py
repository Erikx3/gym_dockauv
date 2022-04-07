import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Used for typehints
from typing import List
import numpy as np

from ..objects.shape import Shape
from .blitmanager import BlitManager


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
    def init_path_animation(self) -> matplotlib.pyplot.axes:
        """
        Initialization of the live animation plot for the path, includes:
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
        # Pause for any initialization to be done
        plt.pause(0.01)
        return self.ax_path

    def add_episode_text(self, ax: matplotlib.pyplot.axes, episode_nr: int) -> None:
        """
        Add episode number top right, needs to be in blit manager to display during live animation

        :param ax: existent axes (of this instance!)
        :param episode_nr: episode nr
        :return: None
        """

        # Add a episode number (need to be added to the blit manager, so it is drawn correctly on each call)
        episode_text = ax.annotate(
            f"Episode: {episode_nr}",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )

        self.bm.add_artists([episode_text])

    def add_shapes(self, ax: matplotlib.pyplot.axes, shapes: List[Shape]) -> None:
        """
        Ass shape as static to existent axes

        :param shapes: list of shapes objects (e.g. Cylinder)
        :param ax: existent axes (of this instance!)
        :return: None
        """
        # Static shapes (assumed for now, drawing these dynamically could slow down the system significantly):
        for shape in shapes:
            ax.plot_surface(*shape.get_plot_variables(), color='r', alpha=1.00)

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

        # Alternative way of older Matplotlib API
        # self.ax_path.path_art.set_data(position[:, :2].T)
        # self.ax_path.path_art.set_3d_properties(position[:, 2])

        # self.ax_path.head_art.set_data(np.array(position[-1, :2].T)[:, None])
        # self.ax_path.head_art.set_3d_properties(np.array(position[-1, 2]))  # Note: Always 1d array fine here

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

    def save_animation(self, save_path: str, frames: int, fps: int = 10, **kwargs) -> None:
        """
        Saves video as mp4 for the animation

        Depending on the amount of subplots etc, necessary arguments need to be given in kwargs.

        :param frames: should be number of time steps for an episode
        :param fps: fps of the saved video, if you want it in real time, enter, 1/dt here, where dt is the step size
        :param save_path: absolute path of where to store the video

        :Keyword Arguments:
            * *position* (``np.ndarray``) --
              array nx3, where n is the total number of all available position data points after the episode

        :return: None
        """

        ani = animation.FuncAnimation(
            self.fig, func=self.save_wrap_update_animation, frames=frames, fargs=(kwargs,))

        writer_video = animation.FFMpegWriter(fps=fps)

        # TODO: Make this a logger statement
        print(f"\nSave video at {save_path}")
        # TODO: Animation so far can not be saved closed onced opened, workaround would be to initialize everything in
        #   a new figure again, but this is a current todo on matplotlib too, should not slow down simulation
        #   significantly, since should not save too many videos anyway
        ani.save(save_path, writer=writer_video)
