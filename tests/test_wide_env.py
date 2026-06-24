import gymnasium as gym
import gymnasium.envs.box2d.car_racing as cr
import pytest

from grass_env import make_carracing
from wide_env import WideViewCarRacing


def test_wide_view_env_is_registered_with_timelimit():
    env = gym.make("WideViewCarRacing-v0", zoom_factor=0.7)
    try:
        assert env.spec.max_episode_steps == 1000
        assert env.spec.reward_threshold == 900
        assert env.unwrapped.zoom_factor == pytest.approx(0.7)
    finally:
        env.close()


def test_make_carracing_uses_wide_env_for_non_default_zoom():
    env = make_carracing(zoom_factor=0.7)()
    try:
        assert env.unwrapped.spec.id == "WideViewCarRacing-v0"
        assert env.unwrapped.zoom_factor == pytest.approx(0.7)
        assert env.observation_space.shape == (96, 96, 1)
    finally:
        env.close()


def test_wide_view_render_scales_module_zoom_and_restores(monkeypatch):
    env = WideViewCarRacing(zoom_factor=0.7)
    original_zoom = cr.ZOOM
    seen = {}

    def fake_render(self, mode):
        seen["zoom"] = cr.ZOOM
        return f"rendered:{mode}"

    monkeypatch.setattr(cr.CarRacing, "_render", fake_render)
    try:
        assert env._render("state_pixels") == "rendered:state_pixels"
        assert seen["zoom"] == pytest.approx(original_zoom * 0.7)
        assert cr.ZOOM == pytest.approx(original_zoom)
    finally:
        env.close()


def test_wide_view_render_restores_zoom_after_error(monkeypatch):
    env = WideViewCarRacing(zoom_factor=0.6)
    original_zoom = cr.ZOOM

    def fake_render(self, mode):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr.CarRacing, "_render", fake_render)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            env._render("state_pixels")
        assert cr.ZOOM == pytest.approx(original_zoom)
    finally:
        env.close()
