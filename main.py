from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import A2C, PPO, DDPG, SAC

from gym_dockauv.config.env_config import TRAIN_CONFIG
import gym_dockauv.train as train
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl
import os

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

GYM_ENV = ["SimpleDocking3d-v0", "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
MODELS = [PPO, SAC]
MODELS_STR = ["_PPO", "_SAC"]
HYPER_PARAMS = [PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST]

if __name__ == "__main__":
    # ---------- TRAINING ----------
    # Training for multiple models and environment at once
    # for GYM in GYM_ENV:
    #     for K, MODEL in enumerate(MODELS):
    #         train.train(gym_env=GYM,
    #                     total_timesteps=1000000,
    #                     MODEL=MODEL,
    #                     model_save_path="logs/"+GYM+MODELS_STR[K],
    #                     agent_hyper_params=HYPER_PARAMS[K],
    #                     env_config=TRAIN_CONFIG,
    #                     tb_log_name=GYM+MODELS_STR[K],
    #                     timesteps_per_save=40000,
    #                     model_load_path=None)

    # Training for one model and one environment
    train.train(gym_env=GYM_ENV[0],
                total_timesteps=50000,
                MODEL=SAC,
                model_save_path="logs/SAC_docking",
                agent_hyper_params=SAC_HYPER_PARAMS_TEST,
                env_config=TRAIN_CONFIG,
                tb_log_name="SAC",
                timesteps_per_save=10000,
                model_load_path=None)
    # Uncomment for plots of previous single run
    train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/logs",
                                  show_full=True, show_episode=False)

    # ---------- PREDICTION ----------
    # Prediction for multiple models and environment at once
    # for subdir, dirs, files in os.walk("/home/erikx3/PycharmProjects/gym_dockauv/rew1_final_model"):
    #     for file in sorted(files):
    #         file_split = file.split("_")
    #         ENV = file_split[0]
    #         model_str = file_split[1]
    #         MODEL = PPO if model_str == "PPO" else SAC
    #         # print(MODEL, ENV)
    #         train.predict(gym_env=ENV, model_path=os.path.join(subdir, file), MODEL=MODEL, n_episodes=1000, render=False)

    # Prediction for one model and one environment
    # train.predict(gym_env=GYM_ENV[0], model_path="logs/SAC_docking_50000", MODEL=SAC, n_episodes=3, render=True)
    # # Uncomment for plots of previous single run
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/predict_logs")

    # ---------- MANUAL ----------
    # Manual flight in an environment
    # train.manual_control(gym_env=GYM_ENV[3])
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/manual_logs")

    # ---------- VIDEO GENERATION ----------
    # Example code on how to save a video of on of the saved episode from either prediction or training
    # epi_stor = EpisodeDataStorage()
    # epi_stor.load(file_name="/home/erikx3/PycharmProjects/gym_dockauv/predict_logs/2022_06_30T17_13_06__Prediction Run__EPISODE_1_DATA_STORAGE.pkl")
    # epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=20)
