import numpy as np
import gym
import gym_dockauv
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# if __name__ == "__main__":
#
#     # Create environment
#     env = gym.make("docking3d-v0")
#
#     # Instantiate the agent
#     model = A2C('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=10000)    # Train the agent
#     # Save the agent
#     model.save("A2C_docking")
#     del model  # delete trained model to demonstrate loading
#
#     # Load the trained agent
#     # NOTE: if you have loading issue, you can pass `print_system_info=True`
#     # to compare the system on which the model was trained vs the current one
#     # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
#     model = A2C.load("A2C_docking", env=env)
#
#     # Evaluate the agent
#     # NOTE: If you use wrappers with your environment that modify rewards,
#     #       this will be reflected here. To evaluate with original rewards,
#     #       wrap environment in a "Monitor" wrapper before other wrappers.
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
#     # Enjoy trained agent
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = env.step(action)
#         #env.render()

if __name__ == "__main__":
    env = gym.make("docking3d-v0")
    done = False
    env.reset()
    for i in range(45):
        while not done:
            # obs, reward, done, info = env.step(env.action_space.sample())
            obs, reward, done, info = env.step(np.array([1, 0, 0, 0, 0, 0]))
            if i % 5 == 0:
                env.render()
        env.reset()
        done = False
