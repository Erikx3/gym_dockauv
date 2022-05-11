import numpy as np
import gym
import gym_dockauv
from stable_baselines3.common.evaluation import evaluate_policy

import os
from gym_dockauv.objects.vehicles.LAUV import LAUV
from gym_dockauv.utils.datastorage import EpisodeDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation

import gym_dockauv.train as train

if __name__ == "__main__":
    train.train(total_timesteps=20000)
    train.predict()
    train.post_analysis_directory()

# Testing Simen vehicle and make video
# if __name__ == "__main__":
#
#     lauv = LAUV()
#     lauv.step_size = 0.02
#     nu_c = np.zeros(6)
#     action = np.array([1, 1, 0])
#     epi_stor = EpisodeDataStorage()
#     epi_stor.set_up_episode_storage("", lauv, lauv.step_size, nu_c, None, title="Test_lauv", episode=123)
#     n_sim = 1000
#     epi_anim = EpisodeAnimation()
#     ax = epi_anim.init_path_animation()
#     epi_anim.add_episode_text(ax, 123)
#     for i in range(1000):
#         lauv.step(action, nu_c)
#         epi_stor.update(nu_c)
#         epi_anim.update_path_animation(positions=epi_stor.positions, attitudes=epi_stor.attitudes)
#
#     save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LAUV.mp4'))
#     epi_anim.save_animation(save_path=save_path, fps=int(1/lauv.step_size),
#                             frames=epi_stor.positions.shape[0],
#                             positions=epi_stor.positions, attitudes=epi_stor.attitudes)


# Random run with on action
# if __name__ == "__main__":
#     env = gym.make("docking3d-v0")
#     done = False
#     env.reset()
#     for i in range(45):
#         while not done:
#             # obs, reward, done, info = env.step(env.action_space.sample())
#             obs, reward, done, info = env.step(np.array([1, 0, 0, 0, 0, 0]))
#             if i % 5 == 0:
#                 env.render()
#         env.reset()
#         done = False

