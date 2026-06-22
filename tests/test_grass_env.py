"""Unit tests for the on-track speed reward (Phase A) in RewardShapingWrapper.

Uses a tiny stub env so the math is verified deterministically without spinning
up Box2D/CarRacing.
"""

import numpy as np
import pytest
from gymnasium import Env, spaces

from grass_env import RewardShapingWrapper


class _Wheel:
    def __init__(self, on_grass):
        # CarRacing marks a wheel on grass when it touches no road tiles.
        self.tiles = set() if on_grass else {"tile"}


class _Hull:
    def __init__(self, vx, vy):
        self.linearVelocity = (vx, vy)


class _Car:
    def __init__(self, wheels_on_grass, vx, vy):
        self.wheels = [_Wheel(i < wheels_on_grass) for i in range(4)]
        self.hull = _Hull(vx, vy)


class StubCarEnv(Env):
    """Minimal env exposing a controllable `car` and a fixed base reward."""

    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)
        self.car = None
        self._base_reward = 0.0

    def set_state(self, wheels_on_grass, vx, vy, base_reward=0.0):
        self.car = _Car(wheels_on_grass, vx, vy)
        self._base_reward = base_reward

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), self._base_reward, False, False, {}


def _step(wrapper, base_env, wheels_on_grass, vx, vy, base_reward=0.0):
    base_env.set_state(wheels_on_grass, vx, vy, base_reward)
    _, reward, _, _, info = wrapper.step(np.zeros(3, dtype=np.float32))
    return reward, info


def test_speed_reward_adds_full_bonus_on_track():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_speed=0.05)
    # All 4 wheels on road, speed = hypot(3, 4) = 5.0
    reward, info = _step(wrapper, base, wheels_on_grass=0, vx=3.0, vy=4.0, base_reward=1.0)
    assert info["speed"] == pytest.approx(5.0)
    # 1.0 + 0.05 * (4/4) * 5.0
    assert reward == pytest.approx(1.0 + 0.25)


def test_speed_reward_scales_with_wheels_on_road():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_speed=0.05)
    # 2 of 4 wheels on grass -> only half the speed bonus
    reward, _ = _step(wrapper, base, wheels_on_grass=2, vx=10.0, vy=0.0, base_reward=0.0)
    assert reward == pytest.approx(0.05 * (2 / 4) * 10.0)


def test_no_speed_bonus_when_fully_off_track():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_speed=0.05)
    reward, _ = _step(wrapper, base, wheels_on_grass=4, vx=10.0, vy=0.0, base_reward=0.0)
    assert reward == pytest.approx(0.0)


def test_disabled_by_default_is_exact_parity():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base)  # all coeffs default to 0
    reward, _ = _step(wrapper, base, wheels_on_grass=0, vx=10.0, vy=10.0, base_reward=0.7)
    assert reward == pytest.approx(0.7)


def test_smoothness_penalty_is_squared_delta():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_smooth=0.1)
    # First step has no previous action -> no penalty yet.
    base.set_state(0, 0.0, 0.0, base_reward=0.0)
    wrapper.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    # Second step: delta = [1, 0, 0] -> squared L2 = 1.0 -> penalty = -0.1
    base.set_state(0, 0.0, 0.0, base_reward=0.0)
    _, reward, _, _, _ = wrapper.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert reward == pytest.approx(-0.1)
    # Third step: delta = [-2, 0, 0] -> squared L2 = 4.0 -> penalty = -0.4
    # (quadratic: doubling the jerk quadruples the penalty)
    base.set_state(0, 0.0, 0.0, base_reward=0.0)
    _, reward, _, _, _ = wrapper.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    assert reward == pytest.approx(-0.4)
