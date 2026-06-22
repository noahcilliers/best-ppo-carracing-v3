"""Spawn geometry — place N cars on the start line, non-overlapping and facing
the right way. Pure math over the generated track: no Box2D body, no rendering,
so it can be reasoned about (and tested) in isolation.

A track entry is ``(alpha, beta, x, y)`` where ``(x, y)`` is a centerline point
and ``beta`` is the road heading at that point. ``(cos beta, sin beta)`` is the
*across-track* (lateral) direction — it is exactly what CarRacing uses to find
the left/right road edges — so offsetting a car along it keeps the car within
the road width.
"""

import math


def compute_spawn_poses(
    track,
    num_agents,
    track_half_width,
    np_random=None,
    lane_frac=0.5,
    row_step=2,
    grid_columns=2,
):
    """Return a list of ``(init_angle, init_x, init_y)`` start poses.

    Cars are arranged two-per-row (left / right of the centerline) with
    successive rows staggered *backward* along the track. Because the track is a
    closed loop, a negative index wraps to the segment just before the start
    line — i.e. cars line up behind the start, as on a real grid — so no two
    cars overlap.

    For ``num_agents == 1`` the single car is centered exactly on ``track[0]``,
    so the environment reduces precisely to single-car CarRacing.

    Args:
        track: list of ``(alpha, beta, x, y)`` tuples (the env's ``self.track``).
        num_agents: number of cars.
        track_half_width: half the road width in world units (env ``TRACK_WIDTH``).
        np_random: optional ``numpy`` Generator. If given, the car->slot mapping
            is shuffled so no car holds a fixed positional advantage across
            resets (the env passes its seeded ``self.np_random``).
        lane_frac: lateral offset as a fraction of the half-width
            (``0.5`` places a car mid-way between centerline and edge).
        row_step: number of track points between successive grid rows.
        grid_columns: number of side-by-side grid slots before starting a new
            row. The default two-column grid preserves the original behavior.
    """
    n_track = len(track)
    if n_track == 0:
        raise ValueError("track is empty")
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")
    if grid_columns < 1:
        raise ValueError("grid_columns must be >= 1")

    # Which grid slot each car takes. Shuffling removes any fixed advantage.
    slots = list(range(num_agents))
    if np_random is not None:
        np_random.shuffle(slots)

    poses = []
    for car_id in range(num_agents):
        slot = slots[car_id]
        row = slot // grid_columns
        column = slot % grid_columns

        idx = (-row * row_step) % n_track
        _, beta, x, y = track[idx]

        # A lone car sits dead-center so the env matches single-car CarRacing.
        if num_agents == 1 or grid_columns == 1:
            lane = 0.0
        elif grid_columns == 2:
            lane = ((column * 2) - 1) * track_half_width * lane_frac
        else:
            side = (column / (grid_columns - 1)) * 2 - 1
            lane = side * track_half_width * lane_frac

        lat_x, lat_y = math.cos(beta), math.sin(beta)
        poses.append((beta, x + lat_x * lane, y + lat_y * lane))
    return poses
