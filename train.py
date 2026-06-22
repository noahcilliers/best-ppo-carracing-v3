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
import os
import gymnasium as gym
import torch.nn as nn
from grass_env import make_carracing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import TailAwareEvalCallback


N_STACK = 4          # number of frames stacked so the agent can perceive motion
ENV_ID = "CarRacing-v3"


def linear_schedule(initial: float):
    """Learning rate that decays linearly from `initial` to 0 over training.

    SB3 calls this with progress_remaining going 1.0 -> 0.0. Annealing the LR to
    zero stops the policy from drifting/destabilizing in the late stage of a run
    (our earlier constant-LR run peaked at 2.8M then regressed)."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial
    return schedule


def build_venv(n_envs: int, seed: int = 0, shaping: dict = None):
    """A vectorized, grayscale, frame-stacked CarRacing env.

    `shaping` is an optional dict of reward-shaping kwargs (k_grass, k_grass_speed,
    k_smooth, grass_terminate_steps, grass_terminate_penalty). Pass None/empty for
    the raw reward (used for the eval env so the eval metric is the true score)."""
    venv = make_vec_env(
        make_carracing(**(shaping or {})),               # CarRacing + shaping + grayscale
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,                       # true parallelism on CPU
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
    p.add_argument("--lr", type=float, default=1e-4,
                   help="initial learning rate (decays linearly to 0)")
    p.add_argument("--k-grass", type=float, default=0.0,
                   help="flat per-wheel per-step grass penalty (0 = off)")
    p.add_argument("--k-grass-speed", type=float, default=0.0,
                   help="velocity-linked grass penalty coefficient (0 = off)")
    p.add_argument("--k-speed", type=float, default=0.0,
                   help="on-track speed REWARD coefficient, +k_speed*(wheels_on_road/4)*speed (0 = off)")
    p.add_argument("--k-smooth", type=float, default=0.0,
                   help="action-smoothness penalty coefficient (0 = off)")
    p.add_argument("--grass-terminate-steps", type=int, default=0,
                   help="terminate after this many consecutive steps with >=3 wheels on grass (0 = off)")
    p.add_argument("--grass-terminate-penalty", type=float, default=0.0,
                   help="reward penalty applied when a grass-termination fires")
    p.add_argument("--target-kl", type=float, default=0.0,
                   help="PPO KL early-stop guardrail (0 = disabled)")
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint .zip to continue training from "
                        "(e.g. checkpoints/ppo_carracing_2000000_steps.zip)")
    args = p.parse_args()

    shaping = dict(
        k_grass=args.k_grass,
        k_grass_speed=args.k_grass_speed,
        k_speed=args.k_speed,
        k_smooth=args.k_smooth,
        grass_terminate_steps=args.grass_terminate_steps,
        grass_terminate_penalty=args.grass_terminate_penalty,
    )
    train_env = build_venv(args.n_envs, seed=args.seed, shaping=shaping)
    # #2 Reward normalization: normalize the RETURN stream (stabilizes value-fn
    # scaling) but NOT the image obs (CnnPolicy already divides by 255). On resume
    # we reload the saved running stats so normalization stays consistent.
    vecnorm_path = "checkpoints/vecnormalize.pkl"
    if args.resume and os.path.exists(vecnorm_path):
        train_env = VecNormalize.load(vecnorm_path, train_env)
    else:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    # Eval env must match the training wrapper stack (VecNormalize) so EvalCallback
    # can sync stats — but with training=False (don't update stats) and
    # norm_reward=False (report the RAW game score as the eval metric).
    eval_env = build_venv(1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    eval_env = VecTransposeImage(eval_env)

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
            learning_rate=linear_schedule(args.lr),  # decays to 0 -> stable endgame
            n_steps=512,            # rollout length PER env  (512 * n_envs per update)
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=(args.target_kl or None),  # KL early-stop guardrail (None = off)
            use_sde=True,           # gSDE: smooth exploration that can't blow up std
            sde_sample_freq=4,      # resample the exploration noise every 4 steps
            policy_kwargs=dict(
                log_std_init=-2.0,          # #1 start exploration NARROW (std ~0.14)
                ortho_init=False,           # #3 RL-Zoo CarRacing recipe knobs
                activation_fn=nn.GELU,
                net_arch=dict(pi=[256], vf=[256]),
            ),
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
    # Tail-aware eval: keep the most RELIABLE checkpoint (best mean-0.5*std over a
    # fixed seed set), not the luckiest 5-episode mean.
    eval_cb = TailAwareEvalCallback(
        eval_env,
        seeds=range(10),                          # fixed seeds -> comparable evals
        eval_freq=max(200_000 // args.n_envs, 1),
        save_path="checkpoints/best/",
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
    train_env.save(vecnorm_path)   # persist reward-normalization stats for resume
    train_env.close()
    eval_env.close()
    print("Done. Final model: checkpoints/ppo_carracing_final.zip")
    print("Best-by-eval model: checkpoints/best/best_model.zip")


if __name__ == "__main__":
    main()
