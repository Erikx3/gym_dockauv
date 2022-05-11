import os

import gym
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO

from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage


def train(total_timesteps: int) -> None:
    """
    Function to train and save agent
    :return: None
    """
    # Create environment
    env = gym.make("docking3d-v0")

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=total_timesteps)    # Train the agent
    # Save the agent
    model.save("PPO_docking")
    env.save()


def predict():
    env = gym.make("docking3d-v0")
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = PPO.load("PPO_docking", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset(seed=5)
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        #print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()


def post_analysis_directory():
    directory = "/home/erikx3/PycharmProjects/gym_dockauv/result_files"

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # Capture full data pkl file
        full_path = os.path.join(directory, filename)
        if filename.endswith("FULL.pkl"):
            full_stor = FullDataStorage()
            full_stor.load(full_path)
            full_stor.plot_rewards()
        elif filename.endswith(".pkl"):
            epi_stor = EpisodeDataStorage()
            epi_stor.load(full_path)
            epi_stor.plot_epsiode_states_and_u()
            epi_stor.plot_rewards()
            plt.show()
            #epi_stor.plot_episode_animation(t_per_step=None, title="Test Post Flight Visualization")