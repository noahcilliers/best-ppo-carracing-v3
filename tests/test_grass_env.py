"""Unit tests for RewardShapingWrapper (squared smoothness penalty, off-by-default
parity).

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
        self.tile_visited_count = 0
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


def test_disabled_by_default_is_exact_parity():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base)  # all coeffs default to 0
    reward, _ = _step(wrapper, base, wheels_on_grass=0, vx=10.0, vy=10.0, base_reward=0.7)
    assert reward == pytest.approx(0.7)


def test_dense_time_cost_is_flat_per_step():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_time=0.05)
    # Flat -k_time every step, on top of the base reward.
    reward, _ = _step(wrapper, base, wheels_on_grass=0, vx=0.0, vy=0.0, base_reward=1.0)
    assert reward == pytest.approx(1.0 - 0.05)


def test_progress_reward_pays_for_new_tiles_only():
    base = StubCarEnv()
    wrapper = RewardShapingWrapper(base, k_progress=2.0)
    base.set_state(0, 0.0, 0.0, base_reward=0.0)
    # Step 1: tile count 0 -> 3, three new tiles -> +2.0 * 3
    base.tile_visited_count = 3
    _, reward, _, _, _ = wrapper.step(np.zeros(3, dtype=np.float32))
    assert reward == pytest.approx(6.0)
    # Step 2: no new tiles (still 3) -> +0  (can't be farmed by dawdling/circling)
    _, reward, _, _, _ = wrapper.step(np.zeros(3, dtype=np.float32))
    assert reward == pytest.approx(0.0)
    # Step 3: 3 -> 5, two new tiles -> +2.0 * 2
    base.tile_visited_count = 5
    _, reward, _, _, _ = wrapper.step(np.zeros(3, dtype=np.float32))
    assert reward == pytest.approx(4.0)


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
