from gym.envs.registration import register


register(
    id='docking3d-v0',
    entry_point='gym_dockauv.envs:Docking3d'
)
