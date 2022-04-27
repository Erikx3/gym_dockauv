# TODO: Train agent

import numpy as np
from gym_dockauv.objects.vehicles.LAUV import LAUV
from gym_dockauv.utils.datastorage import EpisodeDataStorage

import gym_dockauv.utils.geomutils as geom


def train() -> None:
    """
    Function to train and save agent
    :return: None
    """
    print(geom.Tzyx(1, 1, 1))


if __name__ == "__main__":

    lauv = LAUV("/home/erikx3/PycharmProjects/gym_dockauv/gym_dockauv/objects/vehicles/LAUV.xml")
    lauv.step_size = 0.01
    nu_c = np.zeros(6)
    action = np.array([1, 1, 0])
    epi_stor = EpisodeDataStorage()
    epi_stor.set_up_episode_storage("", lauv, lauv.step_size, nu_c, None, title="Test_lauv", episode=123)
    n_sim = 1000
    for i in range(1000):
        lauv.step(action, nu_c)
        epi_stor.update(nu_c)

    epi_stor.plot_episode_animation(None, "LOL")
