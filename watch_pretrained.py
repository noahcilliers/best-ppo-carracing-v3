"""
Watch a pretrained CarRacing agent drive.

Downloads igpaub/ppo-CarRacing-v2 (mean reward ~902, basically "solved") from the
Hugging Face Hub and runs it on the modern CarRacing-v3 env with a live window.

This is purely a pipeline check: if you see a car drive a clean lap, then the
install, the env, the renderer, and the SB3 eval path all work on your Mac.

Compatibility notes handled automatically below:
  * Trained on v2, run on v3 — obs/action spaces are identical, so it transfers.
  * The policy may expect *stacked* frames. We read the model's expected input
    shape and wrap the env with VecFrameStack to match, so there's no crash.

Usage:  .venv/bin/python watch_pretrained.py
"""

import numpy as np
import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

REPO_ID = "igpaub/ppo-CarRacing-v2"
FILENAME = "ppo-CarRacing-v2.zip"
N_EPISODES = 3


def main():
    # --- Download + load the policy ----------------------------------------
    print(f"Downloading {REPO_ID} ...")
    model_path = load_from_hub(repo_id=REPO_ID, filename=FILENAME)
    # custom_objects silences version-mismatch warnings from older SB3 saves.
    model = PPO.load(
        model_path,
        custom_objects={"learning_rate": 0.0, "clip_range": 0.0},
    )
    print("Model loaded. Expected observation shape:", model.observation_space.shape)

    # --- Figure out how many frames the policy expects ----------------------
    # Saved image obs are channel-first (C, 96, 96). The channel dim is the one
    # that isn't 96; stack count = channels / 3 (RGB).
    shape = model.observation_space.shape
    channels = min(shape)            # 3 -> no stack, 6 -> stack 2, 12 -> stack 4
    n_stack = max(1, channels // 3)
    print(f"Using frame stack = {n_stack}")

    # --- Build the env to match, with a live window -------------------------
    def make_env():
        return gym.make("CarRacing-v3", render_mode="human")

    env = DummyVecEnv([make_env])
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)     # HWC -> CHW to match the trained policy

    # --- Drive --------------------------------------------------------------
    for ep in range(N_EPISODES):
        obs = env.reset()
        done = np.array([False])
        total = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += reward[0]
        print(f"Episode {ep + 1}: reward = {total:.1f}")

    env.close()
    print("Done. If the car drove clean laps, the whole pipeline works ✅")


if __name__ == "__main__":
    main()
