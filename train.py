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
from telemetry_env import TelemetryCombinedExtractor, VecTelemetryDict


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


def build_venv(n_envs: int, seed: int = 0, shaping: dict = None,
               zoom_factor: float = 1.0, telemetry: bool = False,
               symmetric_action: bool = False):
    """A vectorized, grayscale, frame-stacked CarRacing env.

    `shaping` is an optional dict of reward-shaping kwargs (k_grass, k_grass_speed,
    k_smooth, grass_terminate_steps, grass_terminate_penalty). Pass None/empty for
    the raw reward (used for the eval env so the eval metric is the true score).
    `zoom_factor` is part of the pixel observation, so train/eval envs must match."""
    venv = make_vec_env(
        make_carracing(                                  # CarRacing + shaping + grayscale
            **(shaping or {}),
            zoom_factor=zoom_factor,
            telemetry=telemetry,
            symmetric_action=symmetric_action,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,                       # true parallelism on CPU
    )
    venv = VecFrameStack(venv, n_stack=N_STACK)          # -> 96x96x4
    if telemetry:
        venv = VecTelemetryDict(venv)                    # -> {"img": stack, "telemetry": current}
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
    p.add_argument("--k-smooth", type=float, default=0.0,
                   help="action-smoothness penalty coefficient (0 = off)")
    p.add_argument("--k-time", type=float, default=0.0,
                   help="dense time cost: extra -k_time per step on top of the base "
                        "-0.1/step, to reward finishing laps faster (0 = off)")
    p.add_argument("--k-progress", type=float, default=0.0,
                   help="forward-progress reward: +k_progress per newly-visited tile "
                        "per step; pairs with --k-time to stay on the racing line (0 = off)")
    p.add_argument("--grass-terminate-steps", type=int, default=0,
                   help="terminate after this many consecutive steps with >=3 wheels on grass (0 = off)")
    p.add_argument("--grass-terminate-penalty", type=float, default=0.0,
                   help="reward penalty applied when a grass-termination fires")
    p.add_argument("--zoom-factor", type=float, default=1.0,
                   help="camera zoom scale (1.0 = stock CarRacing; 0.7 = wider Phase D FOV)")
    p.add_argument("--telemetry", action="store_true",
                   help="use Dict obs with image stack plus current-frame vehicle telemetry")
    p.add_argument("--symmetric-action", action="store_true",
                   help="use 2D (steer, throttle) actions mapped to native gas/brake")
    p.add_argument("--target-kl", type=float, default=0.0,
                   help="PPO KL early-stop guardrail (0 = disabled)")
    p.add_argument("--resume", default=None,
                   help="continue the SAME run from a checkpoint .zip: keeps the "
                        "saved LR schedule and timestep counter (crash recovery)")
    p.add_argument("--warm-start", default=None,
                   help="start a NEW run from existing weights (e.g. fine-tune the "
                        "868 for speed): installs a FRESH linear LR schedule from "
                        "--lr and restarts the progress counter. Use this, not "
                        "--resume, when changing the reward (the checkpoint's saved "
                        "schedule has decayed to ~0 and would freeze learning).")
    args = p.parse_args()

    if args.resume and args.warm_start:
        p.error("use --resume OR --warm-start, not both")
    if args.zoom_factor <= 0.0:
        p.error("--zoom-factor must be positive")
    if (args.telemetry or args.symmetric_action) and args.warm_start:
        p.error("Phase C obs/action-space changes must train from scratch or --resume a matching checkpoint")

    shaping = dict(
        k_grass=args.k_grass,
        k_grass_speed=args.k_grass_speed,
        k_smooth=args.k_smooth,
        k_time=args.k_time,
        k_progress=args.k_progress,
        grass_terminate_steps=args.grass_terminate_steps,
        grass_terminate_penalty=args.grass_terminate_penalty,
    )
    train_env = build_venv(
        args.n_envs,
        seed=args.seed,
        shaping=shaping,
        zoom_factor=args.zoom_factor,
        telemetry=args.telemetry,
        symmetric_action=args.symmetric_action,
    )
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
    eval_env = build_venv(
        1,
        seed=args.seed + 1000,
        zoom_factor=args.zoom_factor,
        telemetry=args.telemetry,
        symmetric_action=args.symmetric_action,
    )
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    eval_env = VecTransposeImage(eval_env)

    if args.resume or args.warm_start:
        source = args.resume or args.warm_start
        load_kwargs = dict(env=train_env, device=args.device, tensorboard_log="runs/")
        if args.warm_start:
            # New objective from existing weights: REPLACE the checkpoint's
            # exhausted LR schedule (decayed to ~0) with a fresh one from --lr, so
            # fine-tuning actually learns. Mirrors selfplay.load_warmstart_ppo.
            print(f"Warm-starting from {args.warm_start} (fresh LR schedule, lr={args.lr})")
            lr_schedule = linear_schedule(args.lr)
            load_kwargs["custom_objects"] = {
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
            }
        else:
            print(f"Resuming from {args.resume}")
        model = PPO.load(source, **load_kwargs)
        if args.warm_start:
            model.learning_rate = lr_schedule
            model.lr_schedule = lr_schedule
            # PPO.load restores the checkpoint's target_kl (None for the 868), so
            # without this a warm-start has NO KL guardrail -- a too-hot LR then
            # blows the policy up (approx_kl in the hundreds) in one update.
            # Honour --target-kl here so fine-tuning can be capped.
            model.target_kl = (args.target_kl or None)
    else:
        # Hyperparameters: a solid, known-decent starting point for CarRacing PPO.
        # These are the knobs you'll tune in Step 6 (lr, ent_coef, n_steps, etc.).
        policy = "MultiInputPolicy" if args.telemetry else "CnnPolicy"
        policy_kwargs = dict(
            log_std_init=-2.0,          # #1 start exploration NARROW (std ~0.14)
            ortho_init=False,           # #3 RL-Zoo CarRacing recipe knobs
            activation_fn=nn.GELU,
            net_arch=dict(pi=[256], vf=[256]),
        )
        if args.telemetry:
            policy_kwargs.update(
                features_extractor_class=TelemetryCombinedExtractor,
                features_extractor_kwargs=dict(
                    cnn_output_dim=256,
                    telemetry_features_dim=32,
                ),
            )

        model = PPO(
            policy,
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
            policy_kwargs=policy_kwargs,
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

    print(f"Training {args.timesteps:,} steps on {args.n_envs} envs, device={args.device}, "
          f"zoom_factor={args.zoom_factor}, telemetry={args.telemetry}, "
          f"symmetric_action={args.symmetric_action}")
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
