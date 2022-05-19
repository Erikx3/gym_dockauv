import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import logging

# Used for typehints
from typing import List, Union
from ..objects.shape import Shape

from .blitmanager import BlitManager
from .geomutils import Rzyx


# Set logger
logger = logging.getLogger(__name__)


class FullVisualization:
    """
    Offers functions with respect to the whole simulation run with multiple episodes.

    Possible function ideas:
    - Reward function, success and other statistics about the agent or its path/decision
    - All available animations in the folder in chronological order
    """
    pass


class EpisodeVisualization:
    """
    This class offers the possibility for a post simulation analysis for each Episode.
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_episode_animation(states: np.ndarray, episode: int = None, shapes: List[Shape] = None,
                               radar_end_pos: np.ndarray = None, t_per_step: float = None, title: str = None) -> None:
        """
        Wrapper to plot interactive animation of the episode animation after it has been saved

        :param states: nx12 array of states
        :param episode: Episode number
        :param shapes: static shapes in plot
        :param radar_end_pos: array(n_data, n_rays, 3) for all radar end pos
        :param title: title for subplot
        :param t_per_step: time between each frame
        :return: None
        """

        epi_anim = EpisodeAnimation()
        ax = epi_anim.init_path_animation()
        epi_anim.add_episode_text(ax, episode)
        if title:
            ax.set(title=title)
        if shapes:
            epi_anim.add_shapes(ax, shapes)
        if radar_end_pos is not None:
            epi_anim.init_radar_animation(n_rays=radar_end_pos.shape[1])

        states = states

        for i in range(states.shape[0]):
            epi_anim.update_path_animation(positions=states[:i + 1, 0:3], attitudes=states[:i + 1, 3:6])
            if radar_end_pos is not None:
                epi_anim.update_radar_animation(pos=states[i, 0:3], end_pos=radar_end_pos[i, :])
            if t_per_step:
                plt.pause(t_per_step)

    @staticmethod
    def plot_episode_states_and_u(states: np.ndarray, nu_c: np.ndarray, u: np.ndarray, step_size: float,
                                  episode: int = None, title: str = None):
        """
        Plot all the episode states and input u in one figure

        :param states: nx12 array
        :param nu_c: nx6 array of current
        :param u: nxa array of actions
        :param step_size: fixed simulation step size
        :param episode: episode number
        :param title: title of figure
        :return: None
        """

        # Get time array
        time_arr = np.arange(len(states[:, 0])) * step_size

        # Init figure
        fig = plt.figure(figsize=(12, 8))
        ax_posxy = fig.add_subplot(3, 2, 1)
        ax_posz = fig.add_subplot(3, 2, 3)
        ax_euler = fig.add_subplot(3, 2, 5)
        ax_nu_c = fig.add_subplot(4, 2, 2)
        ax_vel = fig.add_subplot(4, 2, 4)
        ax_ang = fig.add_subplot(4, 2, 6)
        ax_u = fig.add_subplot(4, 2, 8)

        if episode or title:
            fig.suptitle(f'{title} - Episode {episode}')

        # Position xy plot
        ax_posxy.plot(states[:, 0], states[:, 1], 'g-')
        ax_posxy.invert_xaxis()
        ax_posxy.invert_yaxis()
        ax_posxy.set_title("Position $x$ and $y$")
        ax_posxy.set_xlabel("x [m]")
        ax_posxy.set_ylabel("y [m]")

        # Height plot
        ax_posz.plot(time_arr, states[:, 2], 'g-')
        ax_posz.invert_yaxis()  # Is here the z axis that we invert
        ax_posz.set_title("Position $z$ in NED")
        ax_posz.set_xlabel("t [s]")
        ax_posz.set_ylabel("z [m]")

        # Euler
        ax_euler.plot(time_arr, np.rad2deg(states[:, 3]), 'y-', label=r"Roll $\phi$")
        ax_euler.plot(time_arr, np.rad2deg(states[:, 4]), 'g-', label=r"Pitch $\theta$")
        ax_euler.plot(time_arr, np.rad2deg(states[:, 5]), 'b-', label=r"Yaw $\psi$")
        ax_euler.set_title(r"Euler angles $\Theta=[\phi, \theta, \psi]^T$")
        ax_euler.set_xlabel("t [s]")
        ax_euler.set_ylabel("deg [°]")
        ax_euler.legend()

        # Water current nu_c
        ax_nu_c.plot(time_arr, nu_c[:, 0], 'y-', label="$u_c$")
        ax_nu_c.plot(time_arr, nu_c[:, 1], 'g-', label="$v_c$")
        ax_nu_c.plot(time_arr, nu_c[:, 2], 'b-', label="$w_c$")
        ax_nu_c.set_title("Vel. current $[u_c, v_c, w_c]^T$ in body frame")
        ax_nu_c.set_xlabel("t [s]")
        ax_nu_c.set_ylabel("vel [m/s]")
        ax_nu_c.legend()

        # Linear Velocity
        ax_vel.plot(time_arr, states[:, 6], 'y-', label="$u$")
        ax_vel.plot(time_arr, states[:, 7], 'g-', label="$v$")
        ax_vel.plot(time_arr, states[:, 8], 'b-', label="$w$")
        ax_vel.set_title("Linear velocities $[u, v, w]^T$ in body frame")
        ax_vel.set_xlabel("t [s]")
        ax_vel.set_ylabel("vel [m/s]")
        ax_vel.legend()

        # Angular Velocity
        ax_ang.plot(time_arr, np.rad2deg(states[:, 9]), 'y-', label=r"Roll $p$")
        ax_ang.plot(time_arr, np.rad2deg(states[:, 10]), 'g-', label=r"Pitch $q$")
        ax_ang.plot(time_arr, np.rad2deg(states[:, 11]), 'b-', label=r"Yaw $r$")
        ax_ang.set_title(r"Angular Velocities $\omega=[p, q, r]^T$")
        ax_ang.set_xlabel("t [s]")
        ax_ang.set_ylabel(r"$\omega$ [°/s]")
        ax_ang.legend()

        # Input
        for i in range(u.shape[1]):
            ax_u.plot(time_arr, u[:, i], label=f"Input {i}", linewidth=0.5)
        ax_u.set_title(r"Input $u$")
        ax_u.set_xlabel("t [s]")
        ax_u.set_ylabel(r"u [?]")
        ax_u.legend()

        fig.subplots_adjust(left=0.125, bottom=0.07, right=0.9, top=0.93, wspace=0.2, hspace=0.6)

    @staticmethod
    def plot_rewards(cum_rewards: np.ndarray, rewards: np.ndarray, episode: Union[int, str] = None, title: str = None,
                     x_title: str = "", meta_data_reward: List[str] = None):
        """

        :param cum_rewards: array(n, r) with n data points and r rewards (cumulative)
        :param rewards: array(n, r) with n data points and r rewards
        :param episode: episode number
        :param title: title of figure
        :param x_title: title of x axis (this function is used in two ways)
        :param meta_data_reward: list of strings with short description of each reward
        :return: None
        """
        # Calculate sums
        cum_rewards_sum = np.sum(cum_rewards, axis=1)
        rewards_sum = np.sum(rewards, axis=1)

        # Init figure
        fig = plt.figure(figsize=(12, 8))
        ax_r = fig.add_subplot(2, 1, 1)
        ax_cum = fig.add_subplot(2, 1, 2)
        if episode or title:
            fig.suptitle(f'{title} - Episode {episode}')

        # rewards non cumulative
        for i in range(rewards.shape[1]):
            ax_r.plot(rewards[:, i],
                      label=meta_data_reward[i] if meta_data_reward else f"Reward {i}")
        ax_r.plot(rewards_sum, label="Sum")
        ax_r.set_title("Rewards")
        ax_r.set_xlabel(x_title)
        ax_r.set_ylabel("r")
        ax_r.legend()

        # rewards cumulative
        for i in range(cum_rewards.shape[1]):
            ax_cum.plot(cum_rewards[:, i],
                        label=meta_data_reward[i] if meta_data_reward else f"Cum reward {i}")
        ax_cum.plot(cum_rewards_sum, label="Sum")
        ax_cum.set_title("Cumulative Rewards")
        ax_cum.set_xlabel(x_title)
        ax_cum.set_ylabel("cum r")
        ax_cum.legend()

        fig.tight_layout()


class EpisodeAnimation:
    """
    This function deals with the live animation of an episode end offered live rendering. Example usage is given in
    the test cases.
    """

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

    def init_radar_animation(self, n_rays: int) -> None:
        """
        WIll be added to the path animation, can only be called after init_path_animation

        :param n_rays: number of rays to initialize
        :return:
        """
        self.ax_path.ray_arts = []
        for i in range(n_rays):
            self.ax_path.ray_arts.append(self.ax_path.plot([], [], [], 'r', alpha=1.0, animated=True)[0])

        self.bm.add_artists(self.ax_path.ray_arts)

        # Pause for any initialization to be done
        plt.pause(0.01)

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

        :param shapes: list of shapes objects (e.g. Capsule)
        :param ax: existent axes (of this instance!)
        :return: None
        """
        # Static shapes (assumed for now, drawing these dynamically could slow down the system significantly):
        for shape in shapes:
            for plot_var in shape.get_plot_variables():
                ax.plot_surface(*plot_var, color='b', alpha=1.00)

    def update_radar_animation(self, pos: np.ndarray, end_pos: np.ndarray) -> None:
        """

        :param pos: array(3,) starting position for rays
        :param end_pos: array(n,3) end point for the rays
        :return:
        """
        for i, ray_art in enumerate(self.ax_path.ray_arts):
            ray_art.set_data_3d([pos[0], end_pos[i, 0]], [pos[1], end_pos[i, 1]], [pos[2], end_pos[i, 2]])

    def update_path_animation(self, positions: np.ndarray, attitudes: np.ndarray) -> None:
        """
        Update the path animation plot by updating the according elements

        .. note::

            As for matplotlib==3.5.1, plt.draw() needs to be called inside of this function to properly
            show attitude arrows (with plt.pause() save animation does not work)

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
        # Flip x and z for being NED conform and set axis to correct aspect ratio
        self.ax_path.set_box_aspect([self.ax_path.get_xlim()[0] - self.ax_path.get_xlim()[1],
                                     self.ax_path.get_ylim()[1] - self.ax_path.get_ylim()[0],
                                     self.ax_path.get_zlim()[0] - self.ax_path.get_zlim()[1]])

        plt.draw()

        # This line below lead to failure in saving the animation, for now add it manually to out of scope code
        # plt.pause(0.001)

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
        Function to return the quivers attitude in {n}

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
        if hasattr(self, 'ax_path'):
            if "positions" in kwargs and "attitudes" in kwargs:
                self.update_path_animation(positions=np.array(kwargs["positions"][:step_nr + 1, :]),
                                           attitudes=np.array(kwargs["attitudes"][:step_nr + 1, :]))
            if "positions" in kwargs and "end_pos" in kwargs:
                self.update_radar_animation(pos=np.array(kwargs["positions"][step_nr, :]),
                                            end_pos=np.array(kwargs["end_pos"][step_nr, :]))

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

        writer_video = animation.FFMpegWriter(fps=fps, bitrate=2000)

        # TODO: Animation so far can not be saved closed onced opened, workaround would be to initialize everything in
        #   a new figure again, but this is a current todo on matplotlib too, should not slow down simulation
        #   significantly, since should not save too many videos anyway, since data is saved and can be replayed in
        #   an animation.
        ani.save(save_path, writer=writer_video)
        logger.info(f"Successfully saved EpisodeAnimation at {save_path}")

