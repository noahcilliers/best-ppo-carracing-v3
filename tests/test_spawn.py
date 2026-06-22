import numpy as np
import pytest

from multicar.spawn import compute_spawn_poses
from multicar.tracks import make_indy_oval_track


def _forward(beta):
    return np.array([-np.sin(beta), np.cos(beta)])


def _lateral(beta):
    return np.array([np.cos(beta), np.sin(beta)])


def test_three_column_spawn_places_first_row_side_by_side():
    track = make_indy_oval_track()
    _, beta, x, y = track[0]
    center = np.array([x, y])
    lat = _lateral(beta)

    poses = compute_spawn_poses(
        track,
        num_agents=6,
        track_half_width=12.0,
        lane_frac=0.5,
        row_step=4,
        grid_columns=3,
    )

    first_row_offsets = [
        float(np.dot(np.array([px, py]) - center, lat))
        for _, px, py in poses[:3]
    ]
    assert first_row_offsets == pytest.approx([-6.0, 0.0, 6.0])


def test_three_column_spawn_staggers_second_row_behind_start():
    track = make_indy_oval_track()
    beta0 = track[0][1]
    fwd = _forward(beta0)
    poses = compute_spawn_poses(track, 6, 12.0, row_step=4, grid_columns=3)
    car0 = np.array((poses[0][1], poses[0][2]))
    car3 = np.array((poses[3][1], poses[3][2]))

    assert float(np.dot(car3 - car0, fwd)) < -10.0


def test_two_column_spawn_preserves_original_left_right_offsets():
    track = make_indy_oval_track()
    _, beta, x, y = track[0]
    center = np.array([x, y])
    lat = _lateral(beta)

    poses = compute_spawn_poses(track, num_agents=2, track_half_width=10.0)
    offsets = [
        float(np.dot(np.array([px, py]) - center, lat))
        for _, px, py in poses
    ]
    assert offsets == pytest.approx([-5.0, 5.0])


def test_spawn_columns_must_be_positive():
    with pytest.raises(ValueError, match="grid_columns"):
        compute_spawn_poses(make_indy_oval_track(), 2, 10.0, grid_columns=0)
