"""Pure track geometry helpers for named MultiCarRacing layouts.

The environment ultimately needs the same centerline tuples used by
CarRacing: ``(alpha, beta, x, y)``. ``(x, y)`` is the road centerline, and
``beta`` points across the road; the forward tangent is
``(-sin(beta), cos(beta))``. Keeping this module free of Box2D and pygame makes
custom layouts cheap to test before any physics bodies are created.
"""

from __future__ import annotations

import math
from typing import Iterable

TrackPoint = tuple[float, float, float, float]

RANDOM_LAYOUT = "random"
INDY_OVAL_LAYOUT = "indy_oval"
TRACK_LAYOUTS = (RANDOM_LAYOUT, INDY_OVAL_LAYOUT)
TRACK_DIRECTIONS = ("ccw", "cw")


def make_indy_oval_track(
    *,
    track_scale: float = 1.0,
    detail_step: float = 3.5,
    direction: str = "ccw",
    straight_half_length: float = 205.0,
    short_chute_half_length: float = 46.0,
    corner_radius: float = 44.0,
) -> list[TrackPoint]:
    """Return an Indianapolis-style oval as CarRacing centerline tuples.

    This is intentionally an Indy-style approximation, not an exact
    Indianapolis Motor Speedway survey. The shape has two long straights, two
    short chutes, and four quarter-turn corners, with the start line centered on
    the front straight so wrapped negative track indices place grid rows behind
    the start line instead of inside a corner.
    """

    if track_scale <= 0:
        raise ValueError("track_scale must be > 0")
    if detail_step <= 0:
        raise ValueError("detail_step must be > 0")
    if direction not in TRACK_DIRECTIONS:
        raise ValueError(f"direction must be one of {TRACK_DIRECTIONS}")

    a = straight_half_length * track_scale
    b = short_chute_half_length * track_scale
    r = corner_radius * track_scale

    front_y = -(b + r)
    back_y = b + r
    right_x = a + r
    left_x = -(a + r)

    points: list[tuple[float, float]] = []

    def append_point(x: float, y: float) -> None:
        p = (float(x), float(y))
        if not points or _distance(points[-1], p) > 1e-9:
            points.append(p)

    def append_line(start: tuple[float, float], end: tuple[float, float]) -> None:
        length = _distance(start, end)
        n = max(1, int(math.ceil(length / detail_step)))
        for i in range(n):
            t = i / n
            append_point(
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )

    def append_arc(
        center: tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> None:
        length = abs(end_angle - start_angle) * radius
        n = max(1, int(math.ceil(length / detail_step)))
        for i in range(n):
            t = i / n
            angle = start_angle + (end_angle - start_angle) * t
            append_point(
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )

    # Counterclockwise racing line: start midway down the front straight, drive
    # east, then make four left turns around the loop.
    append_line((0.0, front_y), (a, front_y))
    append_arc((a, -b), r, -math.pi / 2, 0.0)
    append_line((right_x, -b), (right_x, b))
    append_arc((a, b), r, 0.0, math.pi / 2)
    append_line((a, back_y), (-a, back_y))
    append_arc((-a, b), r, math.pi / 2, math.pi)
    append_line((left_x, b), (left_x, -b))
    append_arc((-a, -b), r, math.pi, 3 * math.pi / 2)
    append_line((-a, front_y), (0.0, front_y))

    if direction == "cw":
        points = [points[0], *reversed(points[1:])]

    track = _points_to_track(points)
    validate_track(track, max_gap=detail_step * 1.5)
    return track


def validate_track(
    track: Iterable[TrackPoint],
    *,
    max_gap: float | None = None,
    playfield: float | None = None,
    road_half_width: float = 0.0,
) -> None:
    """Raise ``ValueError`` if a generated centerline is not usable."""

    points = list(track)
    if len(points) < 8:
        raise ValueError("track must contain at least 8 points")

    xy = []
    for idx, point in enumerate(points):
        if len(point) != 4:
            raise ValueError(f"track point {idx} must have 4 values")
        alpha, beta, x, y = point
        values = (alpha, beta, x, y)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"track point {idx} contains a non-finite value")
        xy.append((x, y))
        if playfield is not None:
            bounds = playfield - road_half_width
            if abs(x) > bounds or abs(y) > bounds:
                raise ValueError(f"track point {idx} exceeds playfield bounds")

    for idx in range(len(xy)):
        gap = _distance(xy[idx], xy[(idx + 1) % len(xy)])
        if gap <= 1e-9:
            raise ValueError(f"track segment {idx} has zero length")
        if max_gap is not None and gap > max_gap:
            raise ValueError(f"track segment {idx} gap {gap:.3f} exceeds {max_gap:.3f}")


def _points_to_track(points: list[tuple[float, float]]) -> list[TrackPoint]:
    if len(points) < 8:
        raise ValueError("need at least 8 points to build a track")

    distances = []
    total = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        segment = _distance(point, nxt)
        distances.append(total)
        total += segment
    if total <= 0:
        raise ValueError("track length must be > 0")

    betas = []
    prev_beta = None
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        theta = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
        beta = theta - math.pi / 2
        if prev_beta is not None:
            while beta - prev_beta > math.pi:
                beta -= 2 * math.pi
            while beta - prev_beta < -math.pi:
                beta += 2 * math.pi
        betas.append(beta)
        prev_beta = beta

    track = []
    for i, (x, y) in enumerate(points):
        alpha = distances[i] / total * 2 * math.pi
        track.append((alpha, betas[i], x, y))
    return track


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
