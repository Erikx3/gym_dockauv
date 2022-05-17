import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_dockauv
from stable_baselines3.common.evaluation import evaluate_policy

import os
from gym_dockauv.objects.vehicles.LAUV import LAUV
from gym_dockauv.utils.datastorage import EpisodeDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.config.PPO_hyperparams import PPO_HYPER_PARAMS_TEST

import gym_dockauv.train as train

from stable_baselines3.common.env_checker import check_env
if __name__ == "__main__":
    train.train(total_timesteps=20000, model_save_path="logs/PPO_docking", agent_hyper_params=PPO_HYPER_PARAMS_TEST,
                timesteps_per_save=4000, model_load_path=None)
    # train.train(total_timesteps=400000, model_path="logs/PPO_docking")
    # train.predict()
    train.post_analysis_directory(directory= "/home/erikx3/PycharmProjects/gym_dockauv/logs")

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


# import gym_dockauv.objects.shape as shape
# import time
#
#
# def test_not_vec(l1, ld, cap1, cap2, cap_rad=0.1):
#     for r1, rd in zip(l1, ld):
#         dist = shape.intersec_dist_line_capsule(l1=r1, ld=rd, cap1=cap1, cap2=cap2, cap_rad=cap_rad)
#
#
# def test_vectorized(l1, ld, cap1, cap2, cap_rad=0.1):
#     dist = shape.intersec_dist_line_capsule_vectorized(l1=l1, ld=ld, cap1=cap1, cap2=cap2, cap_rad=cap_rad)


# if __name__ == '__main__':
#     # Init
#     t_not_vec = []
#     t_vec = []
#     jj = 100 # Number of function executions in a run
#     # # Loop over different amount of rays
#     for i in range(1, 100):
#         # Create arrays
#         l1 = np.random.random((i, 3))
#         ld = np.random.random((i, 3))
#         cap1 = np.random.random(3)
#         cap2 = np.random.random(3)
#
#         # Find the sum over xx calls
#         loop = [[], []]  # Index 0, not vec, Index1, vec
#         for j in range(0, jj):
#             t = time.process_time()
#             test_not_vec(l1, ld, cap1, cap2)
#             elapsed_time = time.process_time() - t
#             loop[0].append(elapsed_time)
#
#             t = time.process_time()
#             test_vectorized(l1, ld, cap1, cap2)
#             elapsed_time = time.process_time() - t
#             loop[1].append(elapsed_time)
#
#         t_not_vec.append(sum(loop[0]))
#         t_vec.append(sum(loop[1]))
#
#     plt.figure()
#     plt.plot(t_not_vec, label="Not vectorized")
#     plt.plot(t_vec, label="Vectorized")
#     plt.legend()
#     plt.xlabel("Number of rays")
#     plt.ylabel(f"t [s] over {jj} runs")
#     plt.show()
