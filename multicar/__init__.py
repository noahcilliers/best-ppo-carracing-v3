"""multicar — N-car CarRacing on the current gymnasium / SB3 stack.

Importing this package registers ``MultiCarRacing-v0`` with gymnasium and
exposes the env class plus the three physics-independent helper modules.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from multicar.multi_car_racing import MultiCarRacing
from multicar.tracks import INDY_OVAL_LAYOUT, RANDOM_LAYOUT, TRACK_LAYOUTS

# Register once (importing twice must not raise).
if "MultiCarRacing-v0" not in gym.registry:
    register(
        id="MultiCarRacing-v0",
        entry_point="multicar.multi_car_racing:MultiCarRacing",
        max_episode_steps=1000,
        reward_threshold=900,
    )

__all__ = ["MultiCarRacing", "RANDOM_LAYOUT", "INDY_OVAL_LAYOUT", "TRACK_LAYOUTS"]
