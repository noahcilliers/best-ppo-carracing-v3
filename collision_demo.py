"""Watch a clean car-to-car collision — no trained model needed.

The policy is still unfinished, so two model-driven cars may just wander and
never actually meet. This demo scripts a deliberate rear-end on a roomy, wide
track so you can see exactly what a collision looks like: one car charges into
the back of another and they bounce apart (the bounciness is the --restitution
knob). Pure scripted controls, so it works today.

Usage:
  .venv/bin/python collision_demo.py                       # windowed bounce
  .venv/bin/python collision_demo.py --restitution 0.9     # very bouncy
  .venv/bin/python collision_demo.py --restitution 0.0     # inelastic (they stick)
"""

import argparse

import numpy as np

from multicar.multi_car_racing import MultiCarRacing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restitution", type=float, default=0.6, help="bounciness 0..1")
    p.add_argument("--track-scale", type=float, default=1.3)
    p.add_argument("--track-width-scale", type=float, default=1.8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--headless", action="store_true", help="no window (for CI/testing)")
    args = p.parse_args()

    env = MultiCarRacing(
        num_agents=2,
        render_mode=None if args.headless else "human",
        collisions=True,
        restitution=args.restitution,
        track_scale=args.track_scale,
        track_width_scale=args.track_width_scale,
        random_spawn=False,  # keep the staged positions predictable
    )
    env.reset(seed=args.seed)
    chaser, target = env.cars  # car 0 chases car 1

    # Stage a rear-end: put the target a few lengths ahead of the chaser, both
    # facing the same way, then give the chaser a running start.
    a = chaser.hull.angle
    fwd = np.array([-np.sin(a), np.cos(a)])
    ahead = np.array(chaser.hull.position) + fwd * 9.0  # clear run-up gap
    delta = ahead - np.array(target.hull.position)
    for b in [target.hull] + list(target.wheels):
        b.position = (b.position[0] + delta[0], b.position[1] + delta[1])
        b.linearVelocity = (0, 0)
        b.angularVelocity = 0
    for b in [chaser.hull] + list(chaser.wheels):
        b.linearVelocity = (18 * fwd[0], 18 * fwd[1])

    print(f"Restitution {args.restitution}: chaser (red) rear-ends the target. "
          f"{'Headless' if args.headless else 'Watch the window.'}")

    def fwd_speed(car):
        v = car.hull.linearVelocity
        return float(np.dot([v[0], v[1]], fwd))

    bumped = False
    for _ in range(args.steps):
        # chaser keeps the gas on; target coasts until it gets hit.
        action = np.array([[0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
        env.step(action)
        if not bumped and fwd_speed(target) > 2.0:  # target only moves once struck
            bumped = True
            print(f"  contact! target kicked to {fwd_speed(target):.1f} forward speed")
    env.close()
    print("Done." if bumped else "Done (no contact — try a larger --track-width-scale).")


if __name__ == "__main__":
    main()
