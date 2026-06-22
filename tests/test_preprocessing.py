"""Unit tests for the observation preprocessor — the one component whose bugs
fail silently (a wandering car, no exception), so it is verified in isolation,
away from Box2D/pygame.

Tests assert external behavior through the public interface only: shapes,
dtypes, value ranges, channel ordering, cold-start, rolling order, and per-car
independence. Inputs are fixed synthetic frames so expected outputs are
computable by hand. A solid-grey RGB frame ``(v, v, v)`` maps to grey value
``v`` because the luminosity weights sum to exactly 1.0 — that keeps the
arithmetic in these tests trivial.
"""

import numpy as np
import pytest

from multicar.preprocessing import ObsPreprocessor, to_grayscale

H = W = 96


def gray_frame(value, n=1):
    """n solid-grey RGB frames of the given value, stacked as (n, H, W, 3)."""
    frames = np.full((n, H, W, 3), value, dtype=np.uint8)
    return frames if n > 1 else frames[0]


def test_output_shape_and_dtype():
    pre = ObsPreprocessor(num_agents=3, n_stack=4)
    batch = pre.reset(gray_frame(10, n=3))
    assert batch.shape == (3, 4, H, W)
    assert batch.dtype == np.uint8


def test_values_in_range_not_normalized():
    # A bright frame must stay near 255, NOT be divided down toward [0, 1].
    pre = ObsPreprocessor(num_agents=1, n_stack=4)
    batch = pre.reset(gray_frame(200))
    assert batch.min() >= 0 and batch.max() <= 255
    assert batch.max() == 200  # grey 200 stays 200, proving no /255


def test_grayscale_matches_reference_weights():
    # Independent recomputation via matrix product (different code path than the
    # sum-of-products in to_grayscale) using gymnasium's exact weights.
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    expected = (rgb.astype(np.float64) @ np.array([0.2125, 0.7154, 0.0721])).astype(np.uint8)
    np.testing.assert_array_equal(to_grayscale(rgb), expected)


def test_channel_first_ordering_newest_last():
    # Feed distinct grey values; newest frame must land at the LAST channel.
    pre = ObsPreprocessor(num_agents=1, n_stack=4)
    pre.reset(gray_frame(10))          # buffer: [10,10,10,10]
    pre.observe(gray_frame(20))        # [10,10,10,20]
    pre.observe(gray_frame(30))        # [10,10,20,30]
    batch = pre.observe(gray_frame(40))  # [10,20,30,40]
    # batch[0] is (4, H, W): channel 0 oldest, channel 3 newest
    assert batch[0, 0].max() == 10
    assert batch[0, 1].max() == 20
    assert batch[0, 2].max() == 30
    assert batch[0, 3].max() == 40


def test_cold_start_repeats_first_frame():
    pre = ObsPreprocessor(num_agents=1, n_stack=4)
    batch = pre.reset(gray_frame(77))
    for ch in range(4):
        assert batch[0, ch].max() == 77 and batch[0, ch].min() == 77


def test_rolling_drops_oldest():
    pre = ObsPreprocessor(num_agents=1, n_stack=4)
    pre.reset(gray_frame(0))
    for v in (1, 2, 3, 4):  # fully flush the cold-start zeros out
        batch = pre.observe(gray_frame(v))
    # after 4 observes the stack is exactly the last 4 frames, in order
    assert [int(batch[0, ch].max()) for ch in range(4)] == [1, 2, 3, 4]


def test_per_car_independence_no_leak():
    pre = ObsPreprocessor(num_agents=2, n_stack=4)
    pre.reset(gray_frame(10, n=2))           # both cars start at 10
    obs = np.stack([gray_frame(50), gray_frame(200)], axis=0)
    batch = pre.observe(obs)                 # car0 newest=50, car1 newest=200
    assert batch[0, 3].max() == 50
    assert batch[1, 3].max() == 200
    # the older slots are still the shared cold-start value — no cross-car leak
    assert batch[0, 0].max() == 10 and batch[1, 0].max() == 10


def test_single_agent_accepts_unstacked_frame():
    # When num_agents == 1 the env returns a plain (H, W, 3) frame.
    pre = ObsPreprocessor(num_agents=1, n_stack=4)
    batch = pre.reset(np.full((H, W, 3), 5, dtype=np.uint8))
    assert batch.shape == (1, 4, H, W)


def test_observe_before_reset_raises():
    pre = ObsPreprocessor(num_agents=1)
    with pytest.raises(RuntimeError):
        pre.observe(gray_frame(1))


def test_wrong_car_count_raises():
    pre = ObsPreprocessor(num_agents=2, n_stack=4)
    with pytest.raises(ValueError):
        pre.reset(gray_frame(10, n=3))  # 3 frames for a 2-car preprocessor
