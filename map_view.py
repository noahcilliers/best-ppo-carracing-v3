"""Show or save the full-track overview without loading a policy.

Useful when you just want to inspect the custom layout, start grid, and whole
map camera before running a trained model. By default, a small scripted
controller cruises the cars around the map so the overview is visibly alive.

Usage:
  .venv/bin/python map_view.py --track-layout indy_oval --num-agents 6 \
      --spawn-columns 3 --track-width-scale 2.0
  .venv/bin/python map_view.py --track-layout indy_oval --static
  .venv/bin/python map_view.py --track-layout indy_oval \
      --save /tmp/indy_overview.png --headless
"""

import argparse

import numpy as np
import pygame

from multicar.multi_car_racing import MultiCarRacing
from multicar.tracks import TRACK_DIRECTIONS, TRACK_LAYOUTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-agents", type=int, default=6)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="do not open a window")
    p.add_argument("--save", default=None, help="write one overview PNG after reset")
    p.add_argument("--static", action="store_true", help="park cars instead of cruising")
    p.add_argument("--gas", type=float, default=0.35, help="scripted cruise throttle")
    p.add_argument("--target-speed", type=float, default=26.0, help="scripted cruise speed cap")
    p.add_argument("--lookahead", type=int, default=13, help="track points to steer toward")
    p.add_argument("--track-scale", type=float, default=1.0)
    p.add_argument("--track-width-scale", type=float, default=2.0)
    p.add_argument("--track-layout", default="indy_oval", choices=TRACK_LAYOUTS)
    p.add_argument("--track-direction", default="ccw", choices=TRACK_DIRECTIONS)
    p.add_argument("--spawn-columns", type=int, default=3)
    args = p.parse_args()

    env = MultiCarRacing(
        num_agents=args.num_agents,
        render_mode=None if args.headless else "human_overview",
        track_scale=args.track_scale,
        track_width_scale=args.track_width_scale,
        track_layout=args.track_layout,
        track_direction=args.track_direction,
        spawn_columns=args.spawn_columns,
        random_spawn=False,
    )
    try:
        env.reset(seed=args.seed)
        print(
            f"Track: {env.track_layout} direction={env.track_direction} "
            f"tiles={len(env.track)} spawn_columns={env.spawn_columns}"
        )
        if args.save:
            save_overview_png(env, args.save)
            print(f"Saved overview: {args.save}")

        if not args.headless:
            print("Cruise mode: on" if not args.static else "Cruise mode: off")
            for _ in range(args.steps):
                action = (
                    np.zeros((args.num_agents, 3))
                    if args.static
                    else scripted_cruise_actions(
                        env,
                        gas=args.gas,
                        target_speed=args.target_speed,
                        lookahead=args.lookahead,
                    )
                )
                env.step(action)
    finally:
        env.close()


def scripted_cruise_actions(env, gas=0.35, target_speed=26.0, lookahead=13):
    """Simple centerline-following controls for visual map demos."""
    actions = np.zeros((env.num_agents, 3), dtype=np.float64)
    track_xy = np.array([(x, y) for _, _, x, y in env.track], dtype=np.float64)
    n_track = len(track_xy)

    for car_id, car in enumerate(env.cars):
        pos = np.array(car.hull.position, dtype=np.float64)
        idx = int(np.argmin(np.sum((track_xy - pos) ** 2, axis=1)))
        target = track_xy[(idx + lookahead) % n_track]
        desired = target - pos
        norm = np.linalg.norm(desired)
        if norm > 1e-9:
            desired /= norm

        fwd = np.array([-np.sin(car.hull.angle), np.cos(car.hull.angle)])
        cross = fwd[0] * desired[1] - fwd[1] * desired[0]
        dot = float(np.clip(np.dot(fwd, desired), -1.0, 1.0))
        heading_error = np.arctan2(cross, dot)

        speed = float(np.linalg.norm(car.hull.linearVelocity))
        throttle = gas if speed < target_speed else 0.08
        brake = 0.15 if speed > target_speed * 1.3 else 0.0

        actions[car_id] = [
            -np.clip(1.4 * heading_error, -1.0, 1.0),
            throttle,
            brake,
        ]
    return actions


def save_overview_png(env, path):
    frame = env.render_overview()
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    pygame.image.save(surf, path)


if __name__ == "__main__":
    main()
