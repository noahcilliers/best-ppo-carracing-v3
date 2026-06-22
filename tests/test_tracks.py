import math

import numpy as np
import pytest

from multicar.tracks import make_indy_oval_track, validate_track


def _xy(track):
    return np.array([(x, y) for _, _, x, y in track], dtype=np.float64)


def _segment_lengths(track):
    points = _xy(track)
    nxt = np.roll(points, -1, axis=0)
    return np.linalg.norm(nxt - points, axis=1)


def _forward(beta):
    return np.array([-math.sin(beta), math.cos(beta)])


def test_indy_oval_is_closed_bounded_and_deterministic():
    first = make_indy_oval_track()
    second = make_indy_oval_track()
    assert first == second

    validate_track(first, max_gap=5.25, playfield=333.0, road_half_width=7.0)
    assert len(first) > 300
    assert _segment_lengths(first).max() <= 5.25


def test_indy_oval_starts_mid_front_straight_counterclockwise():
    track = make_indy_oval_track(direction="ccw")
    _, beta, x, y = track[0]

    assert x == pytest.approx(0.0)
    assert y == pytest.approx(-90.0)
    np.testing.assert_allclose(_forward(beta), np.array([1.0, 0.0]), atol=1e-6)


def test_indy_oval_clockwise_keeps_start_line_and_reverses_heading():
    ccw = make_indy_oval_track(direction="ccw")
    cw = make_indy_oval_track(direction="cw")

    assert ccw[0][2:] == cw[0][2:]
    np.testing.assert_allclose(_forward(cw[0][1]), np.array([-1.0, 0.0]), atol=1e-6)


def test_indy_oval_headings_match_next_centerline_segment():
    track = make_indy_oval_track()
    points = _xy(track)

    for i, (_, beta, _, _) in enumerate(track):
        segment = points[(i + 1) % len(points)] - points[i]
        segment = segment / np.linalg.norm(segment)
        assert float(np.dot(_forward(beta), segment)) == pytest.approx(1.0)


def test_indy_oval_scales_dimensions_without_changing_density():
    small = make_indy_oval_track(track_scale=1.0)
    large = make_indy_oval_track(track_scale=1.25)

    assert len(large) > len(small)
    np.testing.assert_allclose(_xy(large)[0], _xy(small)[0] * 1.25)
    assert _segment_lengths(large).max() <= 5.25


def test_validate_track_rejects_bad_geometry():
    with pytest.raises(ValueError, match="at least 8"):
        validate_track([])

    bad = make_indy_oval_track()
    bad[3] = bad[2]
    with pytest.raises(ValueError, match="zero length"):
        validate_track(bad)

    with pytest.raises(ValueError, match="exceeds playfield"):
        validate_track(make_indy_oval_track(), playfield=20.0)
