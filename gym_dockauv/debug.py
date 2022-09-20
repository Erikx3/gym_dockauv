"""
This file is just an accumulation of files for single purpose debugging or plots
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import json

from gym_dockauv.utils.plotutils import plot_function2d, plot_function3d
from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.envs.docking3d import Reward
from gym_dockauv.objects.sensor import Radar
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 13


def debug_log_precision():
    """
    Debug reward function of log (here with delta distance)
    """
    kwargs = {
        "x_goal": 0.01,
        "x_max": BASE_CONFIG["max_dist_from_goal"]
    }
    plot_function2d(f=Reward.log_precision,
                    xlim=[0.01, BASE_CONFIG["max_dist_from_goal"]],
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
    # print(real_r)
    ax = plt.axes()
    plt.colorbar(plt.imshow(image), ax=ax)
    ax.imshow(image,
              extent=[-psi_max * 180 / np.pi, psi_max * 180 / np.pi, -theta_max * 180 / np.pi, theta_max * 180 / np.pi])
    ax.set_ylabel(r"Vertical sensor angle $\theta_r$ [deg]")
    ax.set_xlabel(r"Horizontal sensor angle $\psi_r$ [deg]")


def mov_avg_plot(ax, data, w=10, **kw):
    x = np.arange(data.shape[0])
    series = pd.Series(data)
    q25 = series.rolling(window=w, center=True).quantile(0.25)
    q75 = series.rolling(window=w, center=True).quantile(0.75)
    avg = series.rolling(window=w, center=True).quantile(0.5)
    ax.fill_between(x, q25, q75, alpha=0.5)
    ax.plot(x, avg, **kw)
    ax.margins(x=0)


def get_statistics_prediction(log_dir: str):
    """
    Function to get the statistics in the full data storage of multiple runs
    :param log_dir: folder with logs in subfolders of environment and DRL algo
    :return:
    """
    result_dict = {}
    for subdir, dirs, files in os.walk(log_dir):
        # Expecting subdirectories with specific name
        dirs.sort()
        for file in sorted(files):
            if file.endswith("FULL_DATA_STORAGE.pkl"):
                success, collision, distance = 0.0, 0.0, 0.0
                # Plot cumulative rewards with percentile
                full_stor = FullDataStorage()
                full_stor.load(os.path.join(subdir, file))
                for count, info in enumerate(full_stor.storage["infos"]):
                    success += info["goal_reached"]
                    collision += info["collision"]
                    distance += info["delta_d"]
                # Transform into percentage
                result_dict[subdir] = {
                    "success": success / count,
                    "collision": collision / count,
                    "distance": distance / count / 20  # 20 is delta_d_max
                }
    with open("rew2_result.json", "w") as fp:
        json.dump(result_dict, fp)


def evaluate_training(log_dir: str):
    """
    Function to evaluate the training with episodic reward
    :param log_dir: folder with logs in subfolders of environment and DRL algo
    :return:
    """
    i_plot = 0
    # fig = plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(4, 2, sharey="row", figsize=(12, 12))
    for subdir, dirs, files in os.walk(log_dir):
        # Expecting subdirectories with specific name
        dirs.sort()
        for file in sorted(files):
            if file.endswith("FULL_DATA_STORAGE.pkl"):
                # Plot cumulative rewards with percentile
                ax_tmp = ax[i_plot//2][i_plot%2]
                full_stor = FullDataStorage()
                full_stor.load(os.path.join(subdir, file))
                cum_rewards = full_stor.storage["cum_rewards"]
                cum_rewards_sum = np.sum(cum_rewards, axis=1)
                mov_avg_plot(ax_tmp, cum_rewards_sum, w=50)
                ax_tmp.legend([subdir.split("/")[-1]])
                ax_tmp.grid(True)
                i_plot += 1
    fig.tight_layout()
    plt.savefig('plot.png', dpi=600)


def evaluate_paths(log_dir: str):
    """
    Function to plot the path of  the vehicle in each episode saved in multiple subfolders
    :param log_dir: folder with logs in subfolders of environment and DRL algo
    :return:
    """
    i_plot = 0
    shape_flag = False
    # fig, ax = plt.subplots(4, 2, figsize=(12, 12), subplot_kw=dict(projection='3d'))
    for subdir, dirs, files in os.walk(log_dir):
        # Expecting subdirectories with specific name
        dirs.sort()
        if subdir.endswith("PPO") or subdir.endswith("SAC"):
            # ax_tmp = ax[i_plot // 2][i_plot % 2]
            fig, ax_tmp = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection='3d'))
            ax_tmp.set_title(subdir.split("/")[-1])
            ax_tmp.set_xlabel("North [m]")
            ax_tmp.set_ylabel("East [m]")
            ax_tmp.set_zlabel("Down [m]")
            ax_tmp.view_init(elev=70, azim=70)
            ax_tmp.set_xlim([-15, 15])
            ax_tmp.set_ylim([-15, 15])
            ax_tmp.set_zlim([-12, 12])
            i_plot += 1
        for file in sorted(files):
            if not file.endswith("FULL_DATA_STORAGE.pkl") and file.endswith("DATA_STORAGE.pkl"):
                epi_stor = EpisodeDataStorage()
                epi_stor.load(os.path.join(subdir, file))
                if not shape_flag:
                    shapes = epi_stor.storage["shapes"]
                    for shape in shapes:
                        for plot_var in shape.get_plot_variables():
                            ax_tmp.plot_surface(*plot_var, color="blue", alpha=0.5)
                            shape_flag = True
                # PLot path
                ax_tmp.plot(epi_stor.positions[:, 0], epi_stor.positions[:, 1], epi_stor.positions[:, 2], alpha=0.8, linewidth=3.5)
        plt.savefig(subdir.split("/")[-1] + '.png', dpi=600)
        shape_flag = False


if __name__ == "__main__":
    # debug_log_precision()
    # debug_disc_goal_contraints()
    # debug_cont_goal_constraints_velocity()
    # debug_cont_goal_constraints_heading_navigation()
    # debug_cont_goal_constraints_attitude()
    # debug_cont_goal_constraints_goal_heading()
    # debug_obstacle_avoidance()
    # evaluate_training("/gym_dockauv/reward2_logs")
    # get_statistics_prediction("/gym_dockauv/rew2_predict_logs")
    # evaluate_paths("/gym_dockauv/rew2_predict_logs")
    # plt.show()
    pass
