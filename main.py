from gym_dockauv.config.PPO_hyperparams import PPO_HYPER_PARAMS_TEST
from gym_dockauv.config.env_config import TRAIN_CONFIG
import gym_dockauv.train as train
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12


GYM_ENV = "CapsuleDocking3d-v0"


if __name__ == "__main__":
    # # ---------- TRAINING ----------
    # train.train(gym_env=GYM_ENV,
    #             total_timesteps=600000,
    #             model_save_path="logs/SAC_docking",
    #             agent_hyper_params=PPO_HYPER_PARAMS_TEST,
    #             env_config=TRAIN_CONFIG,
    #             tb_log_name="SAC",
    #             timesteps_per_save=40000,
    #             model_load_path=None)
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/logs",
    #                               show_full=True, show_episode=False)

    # ---------- PREDICTION ----------
    # train.predict(gym_env=GYM_ENV, model_path="logs/SAC_docking_800000", n_episodes=3, render=True)
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/predict_logs")

    # ---------- MANUAL ----------
    train.manual_control(gym_env=GYM_ENV)
    train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/manual_logs")

    # ---------- VIDEO GENERATION ----------
    # epi_stor = EpisodeDataStorage()
    # epi_stor.load(file_name="/home/erikx3/PycharmProjects/gym_dockauv/predict_logs/2022_06_30T17_13_06__Prediction Run__EPISODE_1_DATA_STORAGE.pkl")
    # epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=20)
