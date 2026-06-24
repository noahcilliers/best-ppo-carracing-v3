"""
Watch one of OUR trained CarRacing models drive.

Loads a checkpoint we trained and runs it on CarRacing-v3 with a live window,
using the exact same wrappers train.py uses (grayscale + 4-frame stack), so the
observations match what the policy was trained on.

Usage:
  # watch the best-by-eval model (default)
  .venv/bin/python watch_model.py

  # watch a specific checkpoint
  .venv/bin/python watch_model.py --model checkpoints/ppo_carracing_final.zip
"""

import argparse
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--zoom-factor", type=float, default=1.0,
                   help="camera zoom scale used by the watched model (1.0 = stock)")
    p.add_argument("--telemetry", action="store_true",
                   help="watch a model trained with Dict image+telemetry observations")
    p.add_argument("--symmetric-action", action="store_true",
                   help="watch a model trained with 2D symmetric throttle actions")
    args = p.parse_args()
    if args.zoom_factor <= 0.0:
        p.error("--zoom-factor must be positive")

    model = PPO.load(args.model)
    print(f"Loaded {args.model}")

    env = DummyVecEnv([
        make_carracing(
            render_mode="human",
            zoom_factor=args.zoom_factor,
            telemetry=args.telemetry,
            symmetric_action=args.symmetric_action,
        )
    ])
    env = VecFrameStack(env, n_stack=N_STACK)
    if args.telemetry:
        env = VecTelemetryDict(env)
    env = VecTransposeImage(env)

    for ep in range(args.episodes):
        obs = env.reset()
        done = np.array([False])
        total = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += reward[0]
        print(f"Episode {ep + 1}: reward = {total:.1f}")

    env.close()


if __name__ == "__main__":
    main()
