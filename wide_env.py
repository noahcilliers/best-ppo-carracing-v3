"""Wide field-of-view CarRacing environment.

Gymnasium's CarRacing render zoom is controlled by a module-level ``ZOOM``
constant, not an instance attribute. ``WideViewCarRacing`` temporarily scales
that constant while rendering so the observation and human render share the
same wider camera without copying Gymnasium's full render method.
"""

import gymnasium as gym
import gymnasium.envs.box2d.car_racing as cr
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register


class WideViewCarRacing(CarRacing):
    """CarRacing with a configurable render zoom factor.

    ``zoom_factor=1.0`` matches stock CarRacing. Lower values widen the camera
    field of view by showing more world space inside the same 96x96 observation.
    """

    def __init__(self, *args, zoom_factor=1.0, **kwargs):
        self.zoom_factor = float(zoom_factor)
        if self.zoom_factor <= 0.0:
            raise ValueError("zoom_factor must be positive")
        super().__init__(*args, **kwargs)

    def _render(self, mode: str):
        saved_zoom = cr.ZOOM
        cr.ZOOM = saved_zoom * self.zoom_factor
        try:
            return super()._render(mode)
        finally:
            cr.ZOOM = saved_zoom


if "WideViewCarRacing-v0" not in gym.registry:
    register(
        id="WideViewCarRacing-v0",
        entry_point="wide_env:WideViewCarRacing",
        max_episode_steps=1000,
        reward_threshold=900,
    )


__all__ = ["WideViewCarRacing"]
