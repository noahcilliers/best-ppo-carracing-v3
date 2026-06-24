"""
Tail-aware evaluation for a trained CarRacing model.

Phase A of the variance-reduction plan: stop judging models by a noisy mean and
start measuring the TAIL, because our weakness is occasional catastrophic
off-track episodes (the 740 baseline is 740 +/- 191).

What it does:
  * Runs N episodes on FIXED seeds (reproducible -> comparable across models).
  * Reports mean, std, min, p10, median, and failure-rate (episodes below a
    threshold) -- the failure-rate / p10 / min are the numbers that matter.
  * Optionally dumps per-episode (seed, reward, length) for failure analysis.

Eval always uses the RAW game reward (no grass penalty), so numbers are
comparable to the 740 baseline.

Usage:
  .venv/bin/python evaluate.py --model checkpoints/best/best_model.zip --episodes 50
  .venv/bin/python evaluate.py --model checkpoints/best_740_recipe_v2.zip --episodes 50 --dump eval_740.json
"""

import argparse
import json
import numpy as np
from grass_env import make_carracing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from telemetry_env import VecTelemetryDict

N_STACK = 4   # must match train.py


def build_eval_env(zoom_factor=1.0, telemetry=False, symmetric_action=False):
    env = DummyVecEnv([
        make_carracing(
            k_grass=0.0,
            zoom_factor=zoom_factor,
            telemetry=telemetry,
            symmetric_action=symmetric_action,
        )
    ])  # raw reward
    env = VecFrameStack(env, n_stack=N_STACK)
    if telemetry:
        env = VecTelemetryDict(env)
    env = VecTransposeImage(env)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--zoom-factor", type=float, default=1.0,
                   help="camera zoom scale used by the evaluated model (1.0 = stock)")
    p.add_argument("--telemetry", action="store_true",
                   help="evaluate a model trained with Dict image+telemetry observations")
    p.add_argument("--symmetric-action", action="store_true",
                   help="evaluate a model trained with 2D symmetric throttle actions")
    p.add_argument("--fail-threshold", type=float, default=500.0,
                   help="episodes scoring below this count as failures (tail metric)")
    p.add_argument("--dump", default=None, help="optional path to save per-episode JSON")
    args = p.parse_args()
    if args.zoom_factor <= 0.0:
        p.error("--zoom-factor must be positive")

    model = PPO.load(args.model)
    env = build_eval_env(
        zoom_factor=args.zoom_factor,
        telemetry=args.telemetry,
        symmetric_action=args.symmetric_action,
    )

    per_episode = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        env.seed(seed)                 # fixed seed -> reproducible track
        obs = env.reset()
        done = np.array([False])
        total, steps = 0.0, 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            total += float(r[0])
            steps += 1
        per_episode.append({"seed": seed, "reward": round(total, 1), "length": steps})

    env.close()

    rewards = np.array([e["reward"] for e in per_episode])
    failures = int((rewards < args.fail_threshold).sum())

    print(f"\nModel: {args.model}")
    print(f"Zoom factor: {args.zoom_factor}")
    print(f"Telemetry: {args.telemetry}")
    print(f"Symmetric action: {args.symmetric_action}")
    print(f"Episodes: {args.episodes} (seeds {args.seed_start}..{args.seed_start + args.episodes - 1})")
    print("-" * 52)
    print(f"  mean         {rewards.mean():8.1f}")
    print(f"  std          {rewards.std():8.1f}   <- variance (headline weakness)")
    print(f"  min          {rewards.min():8.1f}   <- catastrophic tail")
    print(f"  p10          {np.percentile(rewards, 10):8.1f}   <- reliability on bad tracks")
    print(f"  median       {np.median(rewards):8.1f}")
    print(f"  max          {rewards.max():8.1f}")
    print(f"  score (m-0.5s){rewards.mean() - 0.5 * rewards.std():7.1f}   <- checkpoint-selection score")
    print(f"  failures     {failures:4d}/{args.episodes}   (reward < {args.fail_threshold:.0f})")
    print("-" * 52)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "zoom_factor": args.zoom_factor,
                    "telemetry": args.telemetry,
                    "symmetric_action": args.symmetric_action,
                    "episodes": per_episode,
                },
                f,
                indent=2,
            )
        print(f"Per-episode results saved to {args.dump}")
        worst = sorted(per_episode, key=lambda e: e["reward"])[:5]
        print("5 worst seeds (hard-turn candidates):",
              ", ".join(f"{e['seed']}({e['reward']:.0f})" for e in worst))


if __name__ == "__main__":
    main()
