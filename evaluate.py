"""
Evaluate a trained CarRacing model — get a hard number, not a vibe.

Runs N episodes headless (no window, fast) with the same wrappers train.py uses,
and reports mean reward +/- std. Use it to compare checkpoints (e.g. 1M vs 4M)
and to decide when Phase 1 is "done" (~900 mean, low std = clean driving).

Usage:
  .venv/bin/python evaluate.py --model checkpoints/best/best_model.zip --episodes 20
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

N_STACK = 4   # must match train.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    model = PPO.load(args.model)

    def make_env():
        env = gym.make("CarRacing-v3")          # no render -> fast
        return gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)

    mean, std = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, deterministic=True
    )
    print(f"{args.model}")
    print(f"Mean reward over {args.episodes} episodes: {mean:.1f} +/- {std:.1f}")
    env.close()


if __name__ == "__main__":
    main()
