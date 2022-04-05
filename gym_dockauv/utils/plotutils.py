import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from blitmanager import BlitManager

from abc import ABC, abstractmethod
from functools import cached_property

from typing import List


class Shape(ABC):
    """
    This is a base class for any shape, should always contain coordinates of position.
    """

    def __init__(self, position: np.ndarray):
        self.position = np.array(position)

    @abstractmethod
    def get_plot_variables(self) -> List[np.ndarray]:
        """
        Function that returns the plot variables for the matplotlib axes.surface_plot function

        :return: return list of 1d arrays for plotting
        """
        pass


class Sphere(Shape):

    def __init__(self, position: np.ndarray, radius: float):
        super().__init__(position)  # Call inherited init functions and then add to it
        self.radius = radius

    def get_plot_variables(self):
        x_c, y_c, z_c = self.plot_shape
        return [self.position[0] + x_c,
                self.position[1] + y_c,
                self.position[2] + z_c]

    @cached_property
    def plot_shape(self):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x_c = self.radius * np.cos(u) * np.sin(v)
        y_c = self.radius * np.sin(u) * np.sin(v)
        z_c = self.radius * np.cos(v)
        return x_c, y_c, z_c


if __name__ == "__main__":

    # Example use with example adopted from:
    # https: // matplotlib.org / stable / gallery / animation / random_walk.html

    np.random.seed(3)
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")


    def random_walk(num_steps, max_step=0.05):
        """Return a 3D random walk as (num_steps, 3) array."""
        start_pos = np.random.random(3)
        steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
        walk = start_pos + np.cumsum(steps, axis=0)
        return walk


    def update(num, walk, line, dot):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
        fr_number.set_text("frame: {j}".format(j=num))
        # Update sphere
        # sphere.position = walk[num, :]

        # Update dot
        dot.set_data(np.array(walk[num, :2].T)[:, None])
        dot.set_3d_properties(np.array(walk[num, 2]))
        return line, dot


    # Data: 100 random walks as (num_steps, 3) arrays
    num_steps = 100
    fps = 10
    walk = random_walk(num_steps)
    sphere = Sphere(walk[0, :], 0.1)

    # Create lines initially without data
    line = ax.plot([], [], [], 'r--', alpha=0.8, animated=True)[0]
    # Create Sphere instance
    surface_sphere = ax.plot_surface(*sphere.get_plot_variables(), color='g', alpha=0.5)
    # Create dot instance
    dot = ax.plot([], [], [], 'b.', alpha=0.8, markersize=10, animated=True)[0]

    # add a frame number
    fr_number = ax.annotate(
        "0",
        (0, 1),
        xycoords="axes fraction",
        xytext=(10, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        animated=True,
    )

    # Setting the axes properties
    ax.set(xlim3d=(0, 1), xlabel='X')
    ax.set(ylim3d=(0, 1), ylabel='Y')
    ax.set(zlim3d=(0, 1), zlabel='Z')

    bm = BlitManager(fig.canvas, [line, fr_number, dot])
    # make sure our window is on the screen and drawn
    plt.show(block=False)
    plt.pause(0.5)

    for j in range(num_steps):
        # update the artists
        update(j, walk, line, dot)

        # Test
        # Setting the axes properties
        #ax.set(xlim3d=(0-j/num_steps, 1+j/num_steps), xlabel='X')
        #ax.set(ylim3d=(0-j/num_steps, 1+j/num_steps), ylabel='Y')
        #ax.set(zlim3d=(0-j/num_steps, 1+j/num_steps), zlabel='Z')

        # tell the blitting manager to do its thing
        bm.update()
        # Optional for slowing down speed
        plt.pause(0.2)

    # here add example of how to save video in a clean way after "live" animating stuff1 (With the saved data!)
    #
    # ani = animation.FuncAnimation(
    #     fig, func=update, frames=num_steps, fargs=(walks, lines), interval=100)
    #
    # writervideo = animation.FFMpegWriter(fps=fps)
    # ani.save("test.mp4", writer=writervideo)

