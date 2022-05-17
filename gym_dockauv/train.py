import os

import gym
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO

from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage


def train(total_timesteps: int, model_path: str = None) -> None:
    """
    Function to train and save model, own wrapper

    :param total_timesteps: total timesteps for training
    :param model_path: path of existing model, use to continue training with that model
    :return: None
    """
    # Create environment
    env = gym.make("docking3d-v0")

    # Instantiate the agent
    if model_path is None:
        # Default:
        model = PPO('MlpPolicy', env)
        # Custom:
        # model = PPO('MlpPolicy', env, learning_rate=0.003, n_steps=2048,
        #             batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        #             clip_range=0.2, verbose=0)
    else:
        model = PPO.load(model_path, env=env)

    model.learn(total_timesteps=total_timesteps)  # Train the agent
    # Save the agent
    model.save("logs/PPO_docking")
    env.save()


def predict():
    env = gym.make("docking3d-v0")
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = PPO.load("logs/PPO_docking", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset(seed=5)
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()


def post_analysis_directory(directory: str = "/home/erikx3/PycharmProjects/gym_dockauv/logs"):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # Capture full data pkl file
        full_path = os.path.join(directory, filename)
        if filename.endswith("FULL_DATA_STORAGE.pkl"):
            full_stor = FullDataStorage()
            full_stor.load(full_path)
            full_stor.plot_rewards()
            plt.show()
        # Episode Data Storage:
        elif filename.endswith(".pkl"):
            epi_stor = EpisodeDataStorage()
            epi_stor.load(full_path)
            epi_stor.plot_epsiode_states_and_u()
            epi_stor.plot_rewards()
            plt.show()
            # epi_stor.plot_episode_animation(t_per_step=None, title="Test Post Flight Visualization")
