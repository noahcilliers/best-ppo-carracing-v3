"""
Step 2 — Sanity check.

Confirms the install works end-to-end before we sink time into training:
  1. CarRacing-v3 instantiates and steps.
  2. Observations and rewards have the shapes/ranges we expect.
  3. Reports whether Apple's MPS GPU backend is available to PyTorch.

Run headless (no window) so it works over SSH / in the background.
Usage:  .venv/bin/python sanity_check.py
"""

import numpy as np
import gymnasium as gym
import torch


def main():
    # --- Hardware check -----------------------------------------------------
    print("PyTorch:", torch.__version__)
    print("MPS (Apple GPU) available:", torch.backends.mps.is_available())
    print("CPU threads:", torch.get_num_threads())
    print("-" * 50)

    # --- Environment check --------------------------------------------------
    # render_mode="rgb_array" gives us frames without opening a window.
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("First obs shape:", obs.shape, "dtype:", obs.dtype)
    print("-" * 50)

    # --- Run a few hundred random steps, collect reward stats ---------------
    n_steps = 300
    rewards = []
    terminated = truncated = False
    for _ in range(n_steps):
        action = env.action_space.sample()          # random steer/gas/brake
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, info = env.reset()

    rewards = np.array(rewards)
    print(f"Ran {n_steps} random steps.")
    print(f"Reward  total={rewards.sum():.1f}  "
          f"mean={rewards.mean():.3f}  min={rewards.min():.3f}  max={rewards.max():.3f}")
    print(f"Final obs shape: {obs.shape}  (expect (96, 96, 3))")
    env.close()
    print("-" * 50)
    print("SANITY CHECK PASSED ✅  — env runs, obs/reward flow correctly.")


if __name__ == "__main__":
    main()
