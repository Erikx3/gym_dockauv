from gym_dockauv.config.PPO_hyperparams import PPO_HYPER_PARAMS_TEST
from gym_dockauv.config.env_config import TRAIN_CONFIG
import gym_dockauv.train as train

GYM_ENV = "SimpleDocking3d-v0"

if __name__ == "__main__":
    # train.train(total_timesteps=280000,
    #             model_save_path="logs/PPO_docking",
    #             agent_hyper_params=PPO_HYPER_PARAMS_TEST,
    #             env_config=TRAIN_CONFIG,
    #             tb_log_name="PPO",
    #             timesteps_per_save=40000,
    #             model_load_path=None)
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/logs",
    #                               show_full=True, show_episode=False)

    # train.predict("first_working_logs/PPO_docking_409600")
    # train.predict("second_working_logs/PPO_docking_286720")
    # train.predict("logs/PPO_docking_286720")
    # train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/predict_logs")

    train.manual_control(gym_env=GYM_ENV)
    train.post_analysis_directory(directory="/home/erikx3/PycharmProjects/gym_dockauv/manual_logs")





