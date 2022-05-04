import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_dockauv
import os


if __name__ == "__main__":
    env = gym.make("docking3d-v0")
    done = False
    env.reset()
    for i in range(45):
        while not done:
            # obs, reward, done, info = env.step(env.action_space.sample())
            obs, reward, done, info = env.step(np.array([1, 0, 0, 0, 0, 0]))
        print(info)
        env.reset()
        done = False
