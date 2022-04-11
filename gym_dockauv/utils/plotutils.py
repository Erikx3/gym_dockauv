import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Used for typehints
from typing import List

from ..objects.shape import Shape
from .blitmanager import BlitManager
from .datastorage import EpisodeDataStorage

from .geomutils import Rzyx


class FullVisualization:
    """
    TODO:
    Offers functions with respect to the whole simulation run with multiple episodes.

    Possible function ideas:
    - Reward function, success and other statistics about the agent
    - All available animations in the folder in chronological order
    """
    pass


class EpisodeVisualization:
    """
    This class offers the possibility for a post simulation analysis for each Episode.
    """
    def __init__(self, episode_data_storage_file_path: str):
        self.data_storage = EpisodeDataStorage.load(episode_data_storage_file_path)

    def plot_episode_states_and_u(self):
        """
        Plot all the episode states and input u in one figure

        :return:
        """
        pass

    def plot_episode_interactive_animation(self):
        """
        Plot interactive animation of the episode animation

        :return:
        """
        pass


# TODO: Think about adding more plots like input, state variables etc
class EpisodeAnimation:

    def __init__(self):
        self.fig = plt.figure()
        self.bm = BlitManager(self.fig.canvas, [])

        # Some predefinitions
        self.ax_path = None

    def __del__(self):
        plt.close(self.fig)

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

        # Add attitude
        self.ax_path.attitude_art = self.ax_path.quiver([], [], [], [], [], [], length=0.1, normalize=True)

        self.bm.add_artists([self.ax_path.path_art, self.ax_path.head_art])

        # Try to fix deep camera angle issues
        self.ax_path.set_proj_type('ortho')
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

    def update_path_animation(self, positions: np.ndarray, attitudes: np.ndarray) -> None:
        """
        Update the path animation plot by updating the according elements

        :param attitudes: array nx3, including the euler angles (fixed to rigid body coord) so far available
        :param positions: array nx3, where n is the number of all available position data points so far in this episode
        :return: None
        """

        # Update path line
        self.ax_path.path_art.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])

        # Update head dot
        self.ax_path.head_art.set_data_3d(positions[-1, 0], positions[-1, 1], positions[-1, 2])

        # Update attitude arrows:
        self.ax_path.attitude_art.remove()
        self.ax_path.attitude_art = self.ax_path.quiver(
            np.full((3,), positions[-1, 0]),
            np.full((3,), positions[-1, 1]),
            np.full((3,), positions[-1, 2]),
            *self.get_quiver_coords_from_attitude(attitudes[-1, :].flatten()),
            length=0.2 * np.linalg.norm([self.ax_path.get_xlim(), self.ax_path.get_ylim(), self.ax_path.get_zlim()]),
            normalize=True, color='y')
        self.ax_path.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax_path, f'get_{a}lim')() for a in 'xyz')])
        plt.pause(0.001)

        # Alternative way of older Matplotlib API
        # self.ax_path.path_art.set_data(position[:, :2].T)
        # self.ax_path.path_art.set_3d_properties(position[:, 2])

        # self.ax_path.head_art.set_data(np.array(position[-1, :2].T)[:, None])
        # self.ax_path.head_art.set_3d_properties(np.array(position[-1, 2]))  # Note: Always 1d array fine here

        # Update animation plot

        self.bm.update()

    @staticmethod
    def get_quiver_coords_from_attitude(attitude: np.ndarray) -> List[np.ndarray]:
        """
        Function to return the attitude

        :param attitude: array 3x1, including the euler angles (fixed to rigid body coord)
        :return: List of arrays for the coordinates of the quiver arrows direction uvw
        """
        u = Rzyx(*attitude).T.dot(np.array([1, 0, 0]))
        v = Rzyx(*attitude).T.dot(np.array([0, 1, 0]))
        w = Rzyx(*attitude).T.dot(np.array([0, 0, 1]))
        return [u, v, w]

    def save_wrap_update_animation(self, step_nr: int, kwargs: dict) -> None:
        """
        Wrapper function to deal with saving animation (done to keep the original update functions, which update
        LIVE, meaning we simulate the available data bei each frame by the step_nr variable

        :param step_nr: actual step number
        :param kwargs: kwargs, as they appear in save_animation()

        :return: None
        """
        if hasattr(self, 'ax_path') and "positions" in kwargs and "attitudes" in kwargs:
            self.update_path_animation(positions=np.array(kwargs["positions"][:step_nr + 1, :]),
                                       attitudes=np.array(kwargs["attitudes"][:step_nr + 1, :]))

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
        #   significantly, since should not save too many videos anyway, since data is saved and can be replayed in
        #   an animation.
        ani.save(save_path, writer=writer_video)
