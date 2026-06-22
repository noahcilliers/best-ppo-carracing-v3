# CarRacing RL Project Record

This document is the working record of the CarRacing RL project: what we set up,
what we tried, what failed, what improved, and what still needs to be decided.
It is intentionally practical. The goal is to preserve the project memory so we
do not re-learn the same lesson twice.

## Current Project Vision

The end goal is not just a solo CarRacing agent. The end goal is multi-car racing:
multiple cars on the same track, visible to each other, with collisions and enough
racecraft to watch them compete.

The project path is:

1. Train a single-car driver on modern Gymnasium `CarRacing-v3`.
2. Use that driver as the foundation for speed/race reward shaping.
3. Move into a modern multi-car environment and warm-start self-play from the
   trained solo policy.

The single-car model is therefore not a dead-end demo. It is the driving base that
the multi-car racing work builds on.

## Environment And Tooling Setup

We chose the modern stack:

- Python 3.11 in `.venv`
- Gymnasium `CarRacing-v3`
- Stable-Baselines3 PPO
- PyTorch
- Box2D / pygame rendering
- TensorBoard logging

We avoided training against the archived `gym_multi_car_racing` stack because it
targets old `gym` and legacy pyglet/OpenGL rendering. Instead, the multi-car work
is being lifted onto the current Gymnasium/pygame CarRacing base.

Useful scripts created so far:

- `sanity_check.py` checks that Gymnasium CarRacing and the local stack run.
- `watch_pretrained.py` loads and watches a pretrained Hugging Face model.
- `train.py` trains our PPO agent.
- `watch_model.py` watches one of our saved checkpoints.
- `evaluate.py` evaluates a checkpoint over multiple episodes.
- `diagnose.py` extracts training metrics from TensorBoard logs.
- `grass_env.py` adds an optional off-track (grass) reward penalty (`--k-grass`).

## Pretrained Baseline

We downloaded and watched a pretrained CarRacing agent from Hugging Face to prove
the full pipeline worked before committing to training.

The pretrained model drove competently but cautiously. This clarified an important
reward-design point: the standard CarRacing reward mostly rewards visiting new
track tiles and only weakly rewards speed through the `-0.1` per-frame penalty.
So a cautious, safe driver is expected under the default reward.

We also discussed the safety risk of loading third-party models: SB3 models are
pickle-based, so untrusted downloads should be treated as code execution risk.

## First Training Run: What Went Wrong

The first major model was trained from scratch with feedforward PPO and a CNN
policy, but with a less stable default-style setup:

- Plain Gaussian exploration
- Constant learning rate around `3e-4`
- No low initial action standard deviation
- No learning-rate decay
- No reward normalization
- Default-ish policy initialization choices

Early results looked promising. The model learned to follow the track and reached
some high-scoring episodes, but it became unstable later in training.

Observed failure pattern:

- Training reward peaked around the middle of the run.
- Later checkpoints regressed.
- The model looked too aggressive and unstable: it would overshoot turns, drift
  off-track, correct too hard, and sometimes spiral.
- Deterministic evaluation was much worse than the best stochastic training
  behavior.
- The action `std` grew instead of shrinking, meaning exploration/control noise
  was getting larger as training progressed.

Diagnosis:

The problem was not simply that the 96x96 camera view was too small. Public
CarRacing agents can score well using the same kind of view. The more direct
problem was policy instability: plain Gaussian exploration encouraged extreme,
clipped, bang-bang controls, and the constant learning rate let the policy drift
late in training.

Important preserved checkpoints from that run:

- `checkpoints/ppo_carracing_100000_steps.zip` — early/near-random reference
- `checkpoints/ppo_carracing_1000000_steps.zip` — first meaningful driver
- `checkpoints/ppo_carracing_2803520_steps.zip` — old peak checkpoint
- `checkpoints/best_2.8M_v1.zip` — permanent safe copy of the old 2.8M checkpoint

The old 2.8M checkpoint was also copied to `checkpoints/best/best_model.zip` at
one point, but that path is expected to be overwritten by new training runs.

Note: after the second run validated the 740 Phase 1 base, these first-run
checkpoints (including `best_2.8M_v1.zip`) were purged to save disk. Only the
740 model and its supporting files are retained — see the second-run results below.

## Research Findings Before The Second Long Run

We paused before committing to another full training run and researched PPO
training stability for CarRacing-like pixel continuous-control tasks.

Key findings:

1. Feedforward PPO can produce a strong CarRacing driver, but public ~900-level
   agents often use recurrent/LSTM PPO. Feedforward PPO is still the better fit
   for now because it keeps the eventual multi-car self-play setup simpler.
2. gSDE is a better exploration mechanism than plain Gaussian noise for this
   continuous-control setting because it produces smoother state-dependent
   exploration.
3. Starting with a low action standard deviation matters. The selected setting is
   `log_std_init=-2`, which starts std around `0.14` rather than near `1.0`.
4. Linear learning-rate decay is important. The earlier run used a constant LR and
   regressed late.
5. Reward normalization should stabilize the value function, while raw image
   observations should not be normalized because the CNN policy already handles
   image scaling.
6. RL-Zoo-style policy knobs are worth adopting: `ortho_init=False`, GELU
   activation, and a 256-wide actor/critic head.

We deliberately did not switch to:

- RecurrentPPO / LSTM yet
- Beta-distribution policy
- Tanh-squashed custom policy
- A new camera field-of-view
- A grass penalty

Those remain possible later, but the current goal is to isolate the PPO stability
fixes first.

## Current Training Configuration

The current `train.py` configuration includes the major stability fixes:

- PPO with `CnnPolicy`
- Grayscale observations
- 4-frame stacking
- 8 parallel environments
- `n_steps=512`
- `batch_size=256`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `max_grad_norm=0.5`
- gSDE enabled with `use_sde=True`
- `sde_sample_freq=4`
- linear learning-rate decay from the initial LR toward zero
- `log_std_init=-2.0`
- reward normalization via `VecNormalize(norm_obs=False, norm_reward=True)`
- `ortho_init=False`
- GELU activation
- actor/critic MLP heads of `[256]`

The current run is intended to test whether the stability fixes solve the previous
std blowup and late regression. Early signs from the current run look better:

- `std` starts around `0.134`
- `std` remains flat early instead of climbing
- early rewards climb quickly
- checkpoints from the long run look qualitatively better than the first run

The key thing to watch is not just peak reward, but whether the policy remains
stable late in training and whether deterministic eval improves instead of
falling behind stochastic behavior.

## Second Training Run: Results (Phase 1 Baseline)

The stability-focused 4M run finished. The recipe fixes worked.

Outcome:

- Deterministic eval: **740.2 +/- 191.1** over 20 episodes (raw game score).
- Mean roughly doubled versus the first run (old peak ~411 deterministic -> 740).
- This reproduces the documented RL-Zoo feedforward figure (~715).
- `std` stayed flat near `0.134` for the whole run instead of climbing, and reward
  rose without the late regression of the first run. The std blowup is solved.

We treat this 740 model as the Phase 1 base.

One bug hit and fixed: with `VecNormalize` on the training env, `EvalCallback`
requires the eval env to also be a `VecNormalize` (here `training=False`,
`norm_reward=False`, so the eval metric stays the raw game score). The first
attempt crashed at the 50k eval; the eval env now mirrors the training wrapper
stack and the rerun completed cleanly.

Best model files after cleanup (intermediate checkpoints were purged to save disk):

- `checkpoints/best_740_recipe_v2.zip` — permanent safe copy of the Phase 1 model
- `checkpoints/best/best_model.zip` — same model (overwritten by future runs)
- `checkpoints/ppo_carracing_final.zip` — 4M final-state backup
- `checkpoints/vecnormalize.pkl` — reward-normalization stats

### Remaining failure mode

The +/-191 variance is driven by occasional off-track episodes on sharp turns.
Observed to be BOTH causes at once:

- (a) it sees the turn but does not brake/react in time (a control/training gap)
- (b) it does not perceive the turn early enough (a field-of-view limit)

Evidence points to (a) as the larger near-term lever: public feedforward agents
reach ~860 at the same or smaller view, so the 740->860 gap is mostly
training/reward, not vision. Widening the camera field-of-view is more motivated
later, in the speed phase, where a faster car genuinely needs more lookahead and
where a retrain happens anyway.

## Phase A: Tail-Aware Evaluation (honest baseline)

Before more training we fixed how we MEASURE, because a 5-episode mean at std ~191
can save a lucky checkpoint. `evaluate.py` now reports mean / std / min / p10 /
failure-rate over a FIXED seed set, and `TailAwareEvalCallback` (`callbacks.py`)
saves the best checkpoint by `mean - 0.5*std` instead of a noisy mean.

Re-judging the "740" model over 50 fixed seeds gave the HONEST baseline (worse than
the lucky 20-episode read):

- mean 657, std 233, median 711, min 152, p10 329, failure-rate 16/50 (32%).

This confirmed the real problem is the TAIL (median 711, but ~1/3 of tracks tank),
not the mean. Worst seeds captured for a hard-turn suite: 5, 19, 12, 43, 13.

## Phase B: Stability + Control Bundle (Phase 1 COMPLETE)

We bundled the cheap, same-direction fixes into ONE run (instead of many isolated
runs) and reserved one-at-a-time testing for the architecture changes. Bundle:

- action-smoothness penalty (`k_smooth=0.01`)
- velocity-linked grass penalty (`k_grass_speed=0.05`) — punishes high-speed
  corner-cutting, forgives slow recovery (avoids a timid car)
- early-termination on sustained grass (`grass_terminate_steps=50`,
  `grass_terminate_penalty=25`)
- a KL guardrail (`target_kl=0.03`) — REMOVED, see below

The `target_kl` lesson: on a from-scratch run it choked learning badly — every
update early-stopped at epoch 0 (1 epoch instead of 10), KL still overshot, and
reward lagged far behind baseline. We aborted and re-ran WITHOUT `target_kl`. The
740 recipe never needed it; it is the one bundle knob that touches the optimizer,
and from-scratch runs do not want it.

Result (4M, same 50 fixed seeds, raw eval) vs the 657 baseline:

| metric | baseline | Phase B |
|---|---|---|
| mean | 657 | 868 |
| std | 233 | 84 |
| min | 152 | 535 |
| p10 | 329 | 800 |
| median | 711 | 904 |
| failures (<500) | 16/50 (32%) | 0/50 (0%) |

The variance problem is SOLVED: zero catastrophic episodes, p10 800, std cut ~64%.
mean 868 edges past the best public FEEDFORWARD CarRacing PPO (~862) — territory we
thought needed an LSTM. Learning was also far faster (shaped `ep_rew_mean` ~880 by
1M vs the baseline's ~347 at 1M), because the shaping gives denser feedback and the
full 10 epochs were restored.

Phase 1 is COMPLETE. Preserved model: `checkpoints/best_868_phaseB_v3.zip`.

Decision: the entire motivation for further solo phases (C/D/E) was this variance
problem, now solved, so more solo tuning is diminishing returns. We move to
Phase 3 (multi-car), warm-starting from the 868 model. Shelved solo ideas are
recorded in `docs/SOLO_TRAINING_FUTURE_WORK.md`.

## Multi-Car Environment Work

We have already created a modern multi-car environment rather than relying on the
archived legacy repo.

Relevant files:

- `multicar/multi_car_racing.py`
- `multicar/rewards.py`
- `multicar/spawn.py`
- `multicar/preprocessing.py`
- `drive_multi.py`
- `collision_demo.py`
- `tests/test_preprocessing.py`

Current multi-car environment characteristics:

- Forked from modern Gymnasium `CarRacing-v3` instead of old `gym`.
- Uses the current pygame-based rendering path.
- Supports N cars sharing one Box2D world.
- Supports per-car observations.
- Supports per-car competitive tile rewards.
- Tags wheels/cars so tile visits are routed to the correct car.
- Includes spawn logic for multiple cars.
- Includes optional car-to-car collisions.
- Has pure helper modules for reward, spawn, and preprocessing logic so those
  pieces can be reasoned about and tested outside the full Box2D environment.

The multi-car env is not yet the final training setup. The likely next step after
we have a stable solo policy is to wrap or adapt the environment for shared-policy
multi-agent training, probably using a PettingZoo/SuperSuit-style bridge or an
equivalent shared-policy interface into SB3.

## Reward Design Notes

The default single-car reward is mostly tile-progress based:

- positive reward for visiting new road tiles
- small time penalty per frame
- large penalty for going far out of bounds

This encourages completing the track reliably, but not necessarily racing fast.

Reward-shaping status:

- Grass/off-track penalty: IMPLEMENTED in `grass_env.py` (`--k-grass`, default
  `0` = original reward). Penalizes each wheel on grass per step (verified: on
  track `-0.1`, fully off `-0.5` at `k_grass=0.1`). Set up as a clean A/B against
  the 740 baseline (eval reports raw score); not yet run.
- Speed/progress shaping for Phase 2: not yet applied.
- Widen the camera field-of-view: deferred to the speed phase, where a faster car
  genuinely needs more lookahead.

Important decision:

Do not stack reward changes on top of optimizer/exploration changes yet. First
prove the PPO stability recipe on the default reward. Once that baseline is
stable, add reward-shaping experiments one at a time.

## Current Mental Model

The CNN is the policy/value network. It turns stacked pixels into:

- an action distribution for steering, gas, and brake
- a value estimate for the critic

PPO is the training algorithm. It collects experience, computes returns and
advantages from rewards, and updates the network weights.

There are 8 parallel environments, but only one shared policy. The 8 environments
collect experience in parallel; the single learner updates one set of weights from
the combined rollout.

The `4,000,000` training budget means 4 million timesteps: 4 million individual
look-act-reward physics ticks across all environments. With 8 envs and
`n_steps=512`, each PPO iteration collects `4096` timesteps before an update.

## Next Steps

Phase 1 is complete (868 mean / 84 std / 0% failures). Moving to Phase 3 (multi-car).

1. (Optional) confirm on fully held-out seeds:

   ```bash
   .venv/bin/python evaluate.py --model checkpoints/best_868_phaseB_v3.zip --episodes 50 --seed-start 50
   ```

2. Scope the multi-car warm-start: wrap `multicar/` as a PettingZoo ParallelEnv,
   bridge to SB3 via SuperSuit for shared-policy self-play, and warm-start every
   car from `checkpoints/best_868_phaseB_v3.zip`.
3. Decide whether to start multi-car with collisions OFF (learn to share the track)
   then enable them, or ON from the start.
4. Shelved solo improvements (Phases C/D/E + new ideas) live in
   `docs/SOLO_TRAINING_FUTURE_WORK.md`.

## Open Questions

- ANSWERED: the gSDE + low-std + reward-normalized run produced a much better
  deterministic eval (740 vs ~411) and a stable `std`. Phase 1 base established.
- ANSWERED: the velocity-linked grass + smoothness + early-termination bundle cut
  variance dramatically (657/233/32% -> 868/84/0%) WITHOUT timidity (mean rose).
  `target_kl` hurt a from-scratch run and was removed.
- ANSWERED: feedforward PPO reached a strong, reliable Phase-3 base (868, 0%
  failures); RecurrentPPO is not needed for now.
- Should the multi-car training begin with collisions disabled, then enable them
  after cars learn to race side-by-side?
- What wrapper/interface should be used for shared-policy multi-agent PPO?

