"""
Phase 1 — Train a CarRacing PPO agent from scratch.

Goal: a car that drives clean, consistent laps (mean reward ~900, low variance).
This is the foundation we reuse for Phase 2 (speed reward shaping) and Phase 3
(multi-car self-play), so we own the weights end-to-end.

Key design choices (the why):
  * GRAYSCALE + FRAME STACK 4 — a single frame is a still photo with no sense of
    speed or drift. Stacking 4 grayscale frames lets the CNN perceive motion,
    which is the difference between a wandering car and a racing one.
  * SubprocVecEnv with N parallel envs — CarRacing is CPU-bound on a Mac, so we
    collect experience from several envs at once to raise throughput.
  * Checkpoints + eval — never lose a long run to a crash, and always keep the
    best-by-eval model separately from the latest.

Usage:
  # quick benchmark to measure YOUR machine's throughput before committing hours
  .venv/bin/python train.py --timesteps 20000 --n-envs 8

  # the real run (resume-friendly via checkpoints)
  .venv/bin/python train.py --timesteps 2000000 --n-envs 8

Watch progress live in another terminal:
  .venv/bin/tensorboard --logdir runs/
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


N_STACK = 4          # number of frames stacked so the agent can perceive motion
ENV_ID = "CarRacing-v3"


def build_venv(n_envs: int, seed: int = 0):
    """A vectorized, grayscale, frame-stacked CarRacing env."""
    venv = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,                       # true parallelism on CPU
        wrapper_class=gym.wrappers.GrayscaleObservation,  # 96x96x3 -> 96x96x1
        wrapper_kwargs={"keep_dim": True},
    )
    venv = VecFrameStack(venv, n_stack=N_STACK)          # -> 96x96x4
    venv = VecMonitor(venv)                              # logs episode reward/length
    return venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="cpu",
                   help="cpu | mps | auto  (MPS is often NOT faster for this small CNN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint .zip to continue training from "
                        "(e.g. checkpoints/ppo_carracing_2000000_steps.zip)")
    args = p.parse_args()

    train_env = build_venv(args.n_envs, seed=args.seed)
    # Match the eval env's wrapper stack to the training env (SB3 auto-applies
    # VecTransposeImage to image obs) so EvalCallback doesn't warn about a type
    # mismatch and evaluates on exactly the same observation format.
    eval_env = VecTransposeImage(build_venv(1, seed=args.seed + 1000))

    if args.resume:
        # Continue a previous run: load the saved weights AND optimizer state,
        # attach the fresh envs, and (below) keep the step counter going.
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=args.device,
                         tensorboard_log="runs/")
    else:
        # Hyperparameters: a solid, known-decent starting point for CarRacing PPO.
        # These are the knobs you'll tune in Step 6 (lr, ent_coef, n_steps, etc.).
        model = PPO(
            "CnnPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=512,            # rollout length PER env  (512 * n_envs per update)
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=args.device,
            tensorboard_log="runs/",
            verbose=1,
            seed=args.seed,
        )

    # Save a checkpoint every ~100k steps (across all envs) so runs are resumable.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path="checkpoints/",
        name_prefix="ppo_carracing",
    )
    # Periodically evaluate and keep the best-performing model separately.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/best/",
        log_path="runs/eval/",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training {args.timesteps:,} steps on {args.n_envs} envs, device={args.device}")
    print("Watch the 'fps' value in the logs — that's your throughput.")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
        # When resuming, keep the global step counter (and TensorBoard curve)
        # going instead of restarting from zero.
        reset_num_timesteps=not bool(args.resume),
    )

    model.save("checkpoints/ppo_carracing_final")
    train_env.close()
    eval_env.close()
    print("Done. Final model: checkpoints/ppo_carracing_final.zip")
    print("Best-by-eval model: checkpoints/best/best_model.zip")


if __name__ == "__main__":
    main()
