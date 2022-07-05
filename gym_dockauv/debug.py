# This file is used to analyze or visualize various aspect of the environment
import matplotlib.pyplot as plt
import numpy as np

from gym_dockauv.utils.plotutils import plot_function2d, plot_function3d
from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.envs.docking3d import Reward
from gym_dockauv.objects.sensor import Radar


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


# Will also take the same parameter for the angular rate
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


def debug_cont_goal_constraints_heading_navigation():
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


def debug_cont_goal_constraints_goal_heading():
    kwargs = {
        "x_des": 0.0,
        "x_max": np.pi,
        "x_exp": 2.0,
        "x_rev": False,
        "delta_d_des": BASE_CONFIG["dist_goal_reached_tol"],
        "delta_d_max": BASE_CONFIG["max_dist_from_goal"],
        "delta_d_exp": 2.0,
        "delta_d_rev": True
    }
    plot_function3d(f=Reward.cont_goal_constraints,
                    xlim=[0.0, np.pi],
                    ylim=[BASE_CONFIG["dist_goal_reached_tol"], BASE_CONFIG["max_dist_from_goal"]],
                    xlabel=r"$\Delta \psi_g [rad]$",
                    ylabel=r"$\Delta d$ [m]",
                    zlabel="r [-]",
                    **kwargs
                    )


def debug_cont_goal_constraints_attitude():
    kwargs = {
        "x_des": 0.0,
        "x_max": np.pi,
        "x_exp": 2.0,
        "x_rev": False,
        "delta_d_des": BASE_CONFIG["dist_goal_reached_tol"],
        "delta_d_max": BASE_CONFIG["max_dist_from_goal"],
        "delta_d_exp": 1,
        "delta_d_rev": True
    }
    plot_function3d(f=Reward.cont_goal_constraints,
                    xlim=[0.0, BASE_CONFIG["max_attitude"]],
                    ylim=[BASE_CONFIG["dist_goal_reached_tol"], BASE_CONFIG["max_dist_from_goal"]],
                    xlabel=r"$Attitude [rad]$",
                    ylabel=r"$\Delta d$ [m]",
                    zlabel="r [-]",
                    **kwargs
                    )


def debug_obstacle_avoidance():
    theta_max = 70 * np.pi / 180
    psi_max = 70 * np.pi / 180
    d_max = 5
    ray_per_deg = 10 * np.pi / 180
    n_side = int(np.ceil(theta_max * 2 / ray_per_deg)) + 1
    radar = Radar(eta=np.zeros(6), freq=1, alpha=theta_max * 2, beta=psi_max * 2,
                  ray_per_deg=ray_per_deg, max_dist=d_max)
    gamma_c = d_max
    epsilon_c = 0.001
    epsilon_oa = 0.01
    n_rays = radar.n_rays
    # Calculate the single reward distribution points by not summing
    beta = Reward.beta_oa(radar.alpha, radar.beta, theta_max, psi_max, epsilon_oa)
    c = Reward.c_oa(radar.intersec_dist / d_max, d_max)
    r = beta / np.maximum((gamma_c * (1 - c)) ** 2, epsilon_c)
    image = r.reshape((n_side, n_side))

    real_r = Reward.obstacle_avoidance(theta_r=radar.alpha, psi_r=radar.beta, d_r=radar.intersec_dist / 12,
                                       theta_max=theta_max, psi_max=psi_max, d_max=d_max, gamma_c=gamma_c,
                                       epsilon_c=epsilon_c, epsilon_oa=epsilon_oa)
    #print(real_r)
    ax = plt.axes()
    plt.colorbar(plt.imshow(image), ax=ax)
    ax.imshow(image,
              extent=[-psi_max * 180 / np.pi, psi_max * 180 / np.pi, -theta_max * 180 / np.pi, theta_max * 180 / np.pi])
    ax.set_ylabel(r"Vertical sensor angle $\theta_r$ [deg]")
    ax.set_xlabel(r"Horizontal sensor angle $\psi_r$ [deg]")


if __name__ == "__main__":
    # debug_log_precision()
    # debug_disc_goal_contraints()
    # debug_cont_goal_constraints_velocity()
    # debug_cont_goal_constraints_heading_navigation()
    # debug_cont_goal_constraints_attitude()
    # debug_cont_goal_constraints_goal_heading()
    debug_obstacle_avoidance()
    plt.show()
