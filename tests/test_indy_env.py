import numpy as np
import pytest


pytest.importorskip("Box2D")
pytest.importorskip("pygame")

from multicar.multi_car_racing import MultiCarRacing
from map_view import scripted_cruise_actions


def test_indy_oval_env_reset_and_step_smoke():
    env = MultiCarRacing(
        num_agents=2,
        track_layout="indy_oval",
        track_width_scale=1.8,
        render_mode=None,
        random_spawn=False,
        max_episode_steps=3,
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (2, 96, 96, 3)
        assert obs.dtype == np.uint8
        assert info["track_layout"] == "indy_oval"
        assert info["track_tiles"] == len(env.track)

        obs, rewards, terminated, truncated, info = env.step(np.zeros((2, 3)))
        assert obs.shape == (2, 96, 96, 3)
        assert rewards.shape == (2,)
        assert terminated is False
        assert truncated is False
        assert info["track_layout"] == "indy_oval"
        assert info["terminated_per_car"].shape == (2,)
    finally:
        env.close()


def test_indy_oval_overview_render_smoke():
    env = MultiCarRacing(
        num_agents=3,
        track_layout="indy_oval",
        track_width_scale=2.0,
        render_mode="rgb_array_overview",
        random_spawn=False,
    )
    try:
        env.reset(seed=0)
        frame = env.render()
        direct_frame = env.render_overview()
        assert frame.shape == (800, 1000, 3)
        assert direct_frame.shape == frame.shape
        assert frame.dtype == np.uint8
        assert direct_frame.dtype == np.uint8
        assert frame.std() > 0
        assert direct_frame.std() > 0
    finally:
        env.close()


def test_map_view_scripted_cruise_moves_cars():
    env = MultiCarRacing(
        num_agents=2,
        track_layout="indy_oval",
        track_width_scale=2.0,
        render_mode=None,
        random_spawn=False,
    )
    try:
        env.reset(seed=0)
        start = np.array([car.hull.position for car in env.cars], dtype=np.float64)
        action = scripted_cruise_actions(env)
        assert action.shape == (2, 3)
        assert np.any(action[:, 1] > 0)

        for _ in range(20):
            env.step(scripted_cruise_actions(env))
        end = np.array([car.hull.position for car in env.cars], dtype=np.float64)
        assert np.linalg.norm(end - start, axis=1).min() > 0.1
    finally:
        env.close()
