# This file is used to analyze or visualize various aspect of the environment
import matplotlib.pyplot as plt
import numpy as np

from gym_dockauv.utils.plotutils import plot_function2d, plot_function3d
from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.envs.docking3d import Reward


def debug_log_precision():
    """
    Debug reward function of log (here with delta distance)
    """
    kwargs = {
        "x_goal": BASE_CONFIG["dist_goal_reached_tol"],
        "x_max": BASE_CONFIG["max_dist_from_goal"]
    }
    plot_function2d(f=Reward.log_precision,
                    xlim=[BASE_CONFIG["dist_goal_reached_tol"], BASE_CONFIG["max_dist_from_goal"]],
                    xlabel=r"$\Delta d$ [m]", ylabel="r [-]", **kwargs)
    plt.grid()


def debug_disc_goal_contraints():
    kwargs = {
        "x_des": BASE_CONFIG["velocity_goal_reached_tol"]
        }
    plot_function2d(Reward.disc_goal_constraints, xlim=[0, BASE_CONFIG["u_max"]],
                    xlabel=r"$||\dot{\mathbf{p}}|| [m/s]$", ylabel="r [-]",
                    **kwargs)
    plt.grid()


def debug_cont_goal_constraints_velocity():
    kwargs = {
        "x_des": BASE_CONFIG["velocity_goal_reached_tol"],
        "x_max": BASE_CONFIG["u_max"],
        "x_exp": 1.0,
        "x_rev": False,
        "delta_d_des": BASE_CONFIG["dist_goal_reached_tol"],
        "delta_d_max": BASE_CONFIG["max_dist_from_goal"],
        "delta_d_exp": 2.0,
        "delta_d_rev": True
    }
    plot_function3d(f=Reward.cont_goal_constraints,
                    xlim=[BASE_CONFIG["velocity_goal_reached_tol"], BASE_CONFIG["u_max"]],
                    ylim=[BASE_CONFIG["dist_goal_reached_tol"], BASE_CONFIG["max_dist_from_goal"]],
                    xlabel=r"$||\dot{\mathbf{p}}|| [m/s]$",
                    ylabel=r"$\Delta d$ [m]",
                    zlabel="r [-]",
                    **kwargs
                    )


def debug_cont_goal_constraints_heading():
    kwargs = {
        "x_des": 0.0,
        "x_max": np.pi,
        "x_exp": 4.0,
        "x_rev": False,
        "delta_d_des": BASE_CONFIG["dist_goal_reached_tol"],
        "delta_d_max": BASE_CONFIG["max_dist_from_goal"],
        "delta_d_exp": 4.0,
        "delta_d_rev": False
    }
    plot_function3d(f=Reward.cont_goal_constraints,
                    xlim=[0.0, np.pi],
                    ylim=[BASE_CONFIG["dist_goal_reached_tol"], BASE_CONFIG["max_dist_from_goal"]],
                    xlabel=r"$\Delta \psi [rad]$",
                    ylabel=r"$\Delta d$ [m]",
                    zlabel="r [-]",
                    **kwargs
                    )


if __name__ == "__main__":
    # debug_log_precision()
    # debug_disc_goal_contraints()
    debug_cont_goal_constraints_velocity()
    debug_cont_goal_constraints_heading()

    plt.show()
