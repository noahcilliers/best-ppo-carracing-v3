"""Drive N cars on one MultiCarRacing track with the trained single-car policy.

The model is a pure function ``(4, 96, 96) uint8 -> (3,) float`` — it has no idea
there are other cars. We load it once and feed every car its own observation,
batched into a single ``model.predict`` call. Each car's raw RGB frame is run
through ObsPreprocessor so it matches training byte-for-byte (grayscale, 4-frame
stack, channel-first). Cars that finish or crash are frozen via
``info["terminated_per_car"]`` so their dead frames stop steering anything.

This harness intentionally does NOT judge driving quality — with an unfinished
model the cars may wander. It verifies the *pipeline*: obs shapes match the
checkpoint, predict succeeds, the env steps, and (optionally) cars collide.

Usage:
  .venv/bin/python drive_multi.py --num-agents 2
  .venv/bin/python drive_multi.py --num-agents 2 --collisions --render human
  .venv/bin/python drive_multi.py --num-agents 1 --steps 300   # parity check
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

import multicar  # noqa: F401  (registers the env / imports the package)
from multicar.multi_car_racing import MultiCarRacing
from multicar.preprocessing import ObsPreprocessor

DEFAULT_MODEL = "checkpoints/best/best_model.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--collisions", action="store_true", help="enable car-to-car contact")
    p.add_argument("--render", default=None, choices=[None, "human"], help="show a tiled window")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading policy: {args.model}")
    model = PPO.load(args.model, device="cpu")
    exp = model.observation_space.shape
    print(f"  policy expects obs {exp}, action {model.action_space.shape}")

    # Channel-first obs is (C, 96, 96); channels = the non-96 dim. Grayscale
    # frames are 1 channel each, so n_stack == channels.
    n_stack = min(exp)
    print(f"  inferred n_stack = {n_stack}")

    env = MultiCarRacing(
        num_agents=args.num_agents,
        render_mode=args.render,
        collisions=args.collisions,
    )
    pre = ObsPreprocessor(num_agents=args.num_agents, n_stack=n_stack)

    obs, _ = env.reset(seed=args.seed)
    batch = pre.reset(obs)
    totals = np.zeros(args.num_agents)

    for step in range(args.steps):
        actions, _ = model.predict(batch, deterministic=True)  # (N, 3)
        obs, rewards, terminated, truncated, info = env.step(actions)
        batch = pre.observe(obs)
        totals += np.atleast_1d(rewards)
        if terminated or truncated:
            break

    done = info.get("terminated_per_car", np.zeros(args.num_agents, bool))
    print(f"\nFinished after {step + 1} steps.")
    for i in range(args.num_agents):
        tag = "finished/crashed" if np.atleast_1d(done)[i] else "still racing"
        print(f"  car {i}: total reward {totals[i]:8.1f}  ({tag})")
    env.close()


if __name__ == "__main__":
    main()
