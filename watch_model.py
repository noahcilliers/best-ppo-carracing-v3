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
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

N_STACK = 4   # must match train.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    model = PPO.load(args.model)
    print(f"Loaded {args.model}")

    def make_env():
        env = gym.make("CarRacing-v3", render_mode="human")
        return gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)
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
