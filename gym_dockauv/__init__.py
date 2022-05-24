from gym.envs.registration import register
from gym_dockauv.config.env_config import REGISTRATION_DICT

for ide, entry_p in REGISTRATION_DICT.items():
    register(
        id=ide,
        entry_point=entry_p
    )
