# Codex Research: Next CarRacing PPO Experiments

## Goal

Improve the next feedforward PPO training phase for the single-car
Gymnasium `CarRacing-v3` agent.

The current Phase 1 baseline is strong but inconsistent:

- Best model: `740.2 +/- 191.1` deterministic reward over 20 episodes.
- Policy family: Stable-Baselines3 PPO, feedforward `CnnPolicy`, NatureCNN.
- Inputs: grayscale `96x96`, 4-frame stack.
- Action space: continuous steer/gas/brake.
- Training: 8 `SubprocVecEnv`s, gSDE, reward normalization, linear LR decay,
  low initial action std, GELU, `net_arch=[256]`, 4M timesteps.

The main weakness is high variance from occasional catastrophic off-track
episodes on sharp turns. Observed causes:

1. The policy sees the turn but brakes/reacts too late.
2. The default camera does not always give enough lookahead.

Grass/off-track reward penalty, wider field-of-view, and RecurrentPPO/LSTM are
already known candidates. This report focuses on other changes, ranked by
expected impact per effort while staying with feedforward PPO where practical.

## Evidence Anchors

- Gymnasium `CarRacing-v3` uses a `96x96` RGB observation, continuous
  steer/gas/brake actions, and a reward of `-0.1` each frame plus `1000/N` for
  new track tiles. It terminates the lap when enough tiles are visited and ends
  far-off-playfield episodes with `-100`.
  Source: [Gymnasium CarRacing docs](https://gymnasium.farama.org/environments/box2d/car_racing/).
- The rendered observation includes bottom indicators for speed, ABS, steering,
  and gyroscope. Those are useful braking/control signals, so cropping the full
  indicator bar is not obviously beneficial.
  Source: [Gymnasium CarRacing docs](https://gymnasium.farama.org/environments/box2d/car_racing/).
- RL-Baselines3-Zoo's current `CarRacing-v3` PPO recipe is very close to the
  current project recipe, but differs in preprocessing: `FrameSkip(skip=2)`,
  resize to `64x64`, `frame_stack=2`, and `batch_size=128`.
  Source: [RL-Zoo PPO hyperparams](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).
- SB3 recommends separate evaluation environments, checking wrappers carefully,
  using multiple runs/seeds for quantitative results, and increasing training
  budget for better performance.
  Source: [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
- SB3 PPO exposes `target_kl` because clipping alone may not prevent large
  policy updates.
  Source: [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
- gSDE was designed to make continuous-control exploration smoother and reduce
  jerky action noise.
  Source: [Smooth Exploration for Robotic Reinforcement Learning](https://arxiv.org/abs/2005.05719).
- Pixel augmentation methods such as RAD and DrQ show that random crop/shift
  style augmentation can improve pixel-based continuous control, although this
  is not native SB3 PPO.
  Sources: [RAD](https://arxiv.org/abs/2004.14990),
  [DrQ](https://arxiv.org/abs/2004.13649),
  [DrQ-v2](https://arxiv.org/abs/2107.09645).
- Beta-distribution PPO addresses bounded continuous action spaces and reported
  improved CarRacing success rate, but requires custom policy/distribution work
  outside normal SB3 PPO.
  Source: [Beta PPO paper](https://arxiv.org/abs/2111.02202).

## Recommended Experiment Order

Run one change at a time against the `740.2 +/- 191.1` baseline. For each run,
evaluate deterministically on at least 50 episodes and record mean, std, min,
p10, lap-finish rate, and off-track failure count. The tail metrics matter more
than mean alone.

### 1. Robust Evaluation And Checkpoint Selection

**What it does**

Select checkpoints by reliability, not just noisy 5-episode mean reward.

**Why it should help**

With evaluation std near 191, `EvalCallback(n_eval_episodes=5)` can easily save
a lucky checkpoint. The current problem is a tail-risk problem, so checkpoint
selection should explicitly prefer low failure rate and good lower-tail reward.

**Implementation**

Immediate low-code version:

- Increase eval episodes from `5` to `20` or `50`.
- Evaluate every `100k` steps instead of `50k` if needed to keep wall-clock cost
  reasonable.
- After training, run:

```bash
.venv/bin/python evaluate.py --model checkpoints/best/best_model.zip --episodes 50
.venv/bin/python evaluate.py --model checkpoints/ppo_carracing_final.zip --episodes 50
```

Better version:

- Add a custom eval callback that saves by one of:
  - `score = mean - 0.5 * std`
  - p10 reward
  - lowest failure count, then highest mean
- Also persist per-episode rewards and seeds for failure analysis.

**Expected effect**

- Mean: small to medium improvement from avoiding lucky checkpoints.
- Variance: high improvement in selected model reliability.

**Cost/risk**

Low. It does not change training dynamics.

**Multi-car relevance**

Very high. Multi-car self-play should start from the most reliable solo driver,
not the luckiest short-eval checkpoint.

### 2. Add PPO `target_kl`

**What it does**

Stops an update early when the new policy moves too far from the old policy.

**Why it should help**

The first failed training run showed late instability. The current recipe fixed
the largest issue, but occasional oversized PPO updates can still create brittle
behavior. SB3 includes `target_kl` specifically because PPO clipping can be
insufficient by itself.

**Implementation**

Add to `PPO(...)`:

```python
target_kl=0.03,
```

A/B values:

- Start with `0.03`.
- Try `0.05` if learning slows too much.
- Try `0.02` if update metrics remain volatile.

Keep `clip_range=0.2` fixed for the first run.

**Expected effect**

- Mean: neutral to medium improvement.
- Variance: medium reduction through fewer destabilizing updates.

**Cost/risk**

Very low. The main risk is slower learning if the KL limit is too tight.

**Multi-car relevance**

High. PPO update stability becomes more important once opponent behavior makes
the data distribution less stationary.

### 3. Failure-Seed / Hard-Turn Fine-Tuning Curriculum

**What it does**

Fine-tunes the current good policy on a mixture of normal random tracks and
known failure tracks, especially tracks with sharp turns.

**Why it should help**

The agent does not need broad relearning; it needs more experience on the rare
failure modes that dominate variance. Curriculum learning is a standard RL tool
for sequencing task difficulty and reusing experience from easier tasks.

**Implementation**

First add logging:

- During evaluation, record episode seed, reward, length, lap_finished,
  min reward, and optionally a short failure tag.
- Save seeds for episodes below a threshold, e.g. reward `<500` or
  off-track termination.

Then fine-tune:

- Start from `checkpoints/best_740_recipe_v2.zip`.
- During reset, sample from known failure seeds 50-70% of the time and fresh
  random seeds for the rest.
- Train for `1M-2M` steps with low LR, e.g. `3e-5 -> 0`.
- Final selection still uses normal random evaluation, not the hard-seed mix.

Suggested phases:

1. `500k` steps with 70% failure seeds, 30% random.
2. `500k-1.5M` steps with 30% failure seeds, 70% random.
3. Evaluate only on normal random tracks.

**Expected effect**

- Mean: medium improvement.
- Variance: high reduction if failures are repeatable by seed.

**Cost/risk**

Medium. The risk is overfitting to a small seed set. Mitigate by mixing random
tracks and tracking random-eval performance.

**Multi-car relevance**

Very high. The same pattern can later be used for collision, traffic, and
side-by-side racing failures.

### 4. Edge-Margin Penalty Instead Of Full Centerline Reward

**What it does**

Penalizes being near or beyond the road edge, but does not force the car to hug
the exact centerline.

**Why it should help**

The observed failures are off-track catastrophes. A full centerline/lane-keeping
reward may make the driver timid and harm racing lines. A margin penalty gives
an earlier warning before grass without strongly shaping normal driving.

**Implementation**

Use `env.unwrapped.track` and the car hull position to compute distance to the
nearest track segment or nearest centerline point. In Gymnasium CarRacing,
road tiles are built from centerline points plus/minus `TRACK_WIDTH`, so
`TRACK_WIDTH` is effectively the road half-width.

Penalty sketch:

```python
margin = abs(lateral_offset) / road_half_width
excess = max(0.0, margin - 0.75)
reward -= k_edge * excess * excess
```

A/B values:

- `k_edge=0.03`
- `k_edge=0.06`
- `k_edge=0.10`

Log `lateral_margin` and `edge_penalty` to make reward hacking visible.

**Expected effect**

- Mean: medium improvement if crashes drop.
- Variance: high reduction if edge excursions precede failures.

**Cost/risk**

Medium. Too much center/edge shaping can discourage legitimate racing lines,
especially in multi-car overtaking. Keep the penalty inactive until near the
edge.

**Multi-car relevance**

Mixed but useful. It improves solo safety, but do not overfit to centerline
obedience if the eventual race policy needs to pass or recover after contact.

### 5. Small Steering-Rate / Action-Smoothness Penalty

**What it does**

Penalizes abrupt steering changes to reduce late overcorrection and unstable
oscillation.

**Why it should help**

The policy already uses gSDE for smoother exploration, but deterministic control
can still be jerky. Sharp-turn failures often involve delayed correction,
overshoot, and violent counter-correction.

**Implementation**

In a reward wrapper, store the previous action:

```python
delta_steer = action[0] - self.prev_action[0]
reward -= k_delta_steer * abs(delta_steer)
```

A/B values:

- `k_delta_steer=0.01`
- `k_delta_steer=0.02`
- `k_delta_steer=0.03`

Optional later term:

```python
reward -= k_gas_turn * action[1] * abs(action[0])
```

Avoid penalizing brake onset at first. The problem statement says one failure
mode is not braking early enough; a brake-change penalty could make that worse.

**Expected effect**

- Mean: small to medium improvement.
- Variance: medium reduction.

**Cost/risk**

Low to medium. Too strong a smoothness penalty can prevent emergency steering.

**Multi-car relevance**

High. Smooth controls reduce chaotic collisions and make self-play easier to
stabilize.

### 6. A/B The Remaining RL-Zoo Deltas

**What it does**

Tests the few RL-Zoo recipe differences not currently in the baseline.

**Why it should help**

RL-Zoo's `CarRacing-v3` PPO recipe is the closest known empirical reference for
the current stack. The baseline already adopted the major policy/hyperparameter
settings, so the remaining deltas are worth isolating.

**Implementation**

Test in this order:

1. `batch_size=128` instead of `256`.
2. Add `ResizeObservation(..., shape=(64, 64))`, still no frame-skip.
3. Pair `FrameSkip(skip=2)` with `frame_stack=2`, matching RL-Zoo.

Do not combine all three immediately unless the goal is simply to reproduce the
RL-Zoo recipe.

**Expected effect**

- Mean: uncertain to medium improvement.
- Variance: possible reduction, especially if frame-skip/action-repeat smooths
  behavior.

**Cost/risk**

Low for `batch_size=128` and resize. Medium for frame-skip because action-repeat
can reduce reaction time and worsen late braking.

**Multi-car relevance**

Good. If `64x64` preserves performance, it may save compute in multi-car where
per-car rendering cost scales with agent count.

### 7. Longer Low-LR Fine-Tune From The Best Baseline

**What it does**

Continues training from the current best model with a smaller learning rate.

**Why it should help**

The 4M run was stable, and SB3 notes that increasing training budget often
improves model-free RL performance. A low-LR extension may extract remaining
feedforward gains without destabilizing the policy.

**Implementation**

Resume from `checkpoints/best_740_recipe_v2.zip`:

```bash
.venv/bin/python train.py \
  --resume checkpoints/best_740_recipe_v2.zip \
  --timesteps 1000000 \
  --lr 3e-5 \
  --n-envs 8
```

Pair with:

- `target_kl=0.03`
- robust 20-50 episode eval
- same reward as baseline for the first run

**Expected effect**

- Mean: medium improvement.
- Variance: neutral to lower if checkpoint selection is robust.

**Cost/risk**

Low to medium. Compute cost is real, and late overfitting/regression is possible
without the selection changes above.

**Multi-car relevance**

High. Better base weights transfer directly.

### 8. Early Termination On Sustained Grass

**What it does**

Terminates episodes when the car spends too long mostly on grass.

**Why it should help**

The current grass penalty is already implemented and should be tested first.
Early termination is a stronger variant that saves training time on unrecoverable
off-track spirals and makes sustained grass clearly bad.

**Implementation**

In the existing grass wrapper, track consecutive grass steps:

- If `>=3` wheels are on grass for `25-50` consecutive steps, terminate.
- Add a final penalty, e.g. `-25` or `-50`.
- Do not terminate brief edge touches or two-wheel curb events.

Suggested A/B after basic grass penalty:

- `grass_patience=50`, `terminal_penalty=25`
- `grass_patience=25`, `terminal_penalty=25`

**Expected effect**

- Mean: small to medium improvement.
- Variance: medium reduction.

**Cost/risk**

Medium. It may reduce recovery skill, which matters later when contact pushes a
car off line.

**Multi-car relevance**

Mixed. Useful for solo safety, but too harsh for multi-car racing where recovery
from contact is valuable.

### 9. Progress/Speed Shaping After Variance Drops

**What it does**

Adds a reward for faster progress or a stronger time penalty.

**Why it should help**

The default reward already rewards finishing sooner through `-0.1/frame`. Extra
speed shaping can push mean score upward, but it can also amplify the exact
sharp-turn crashes causing high variance.

**Implementation**

Prefer progress-based shaping over raw speed:

- Compute change in nearest track index or arc-length progress.
- Reward forward progress per step.
- Penalize reverse/no-progress behavior.

Initial values should be small:

```python
reward += k_progress * forward_progress_delta
```

Alternatively, slightly increase the time penalty only after the driver is
consistent.

**Expected effect**

- Mean: medium to high improvement if the driver is already reliable.
- Variance: risk of increasing variance if applied too early.

**Cost/risk**

Medium to high. Reward hacking and aggressive driving are likely failure modes.

**Multi-car relevance**

High later. Racing ultimately needs progress-per-time incentives, but it should
come after the solo policy is reliably safe.

### 10. Wider Or Separate Visual Feature Extractor

**What it does**

Gives the CNN/value/policy more capacity or lets actor and critic use separate
feature extractors.

**Why it should help**

NatureCNN is generic and shared between actor and critic by default. More
capacity may improve turn perception and value estimates on hard tracks.

**Implementation**

Low-risk first try:

```python
policy_kwargs=dict(
    log_std_init=-2.0,
    ortho_init=False,
    activation_fn=nn.GELU,
    net_arch=dict(pi=[256], vf=[256]),
    share_features_extractor=False,
)
```

Later:

- Custom CNN with `features_dim=768` or `1024`.
- Slightly deeper conv stack.

**Expected effect**

- Mean: small to medium improvement.
- Variance: uncertain.

**Cost/risk**

Medium to high on Apple CPU. More capacity can slow training and overfit.

**Multi-car relevance**

Good if compute allows. Multi-car observations will include cars/opponents,
which may require more visual capacity.

### 11. Pixel Augmentation

**What it does**

Applies random shifts/crops or similar image augmentation during training.

**Why it should help**

RAD and DrQ-style methods improve robustness in pixel-based continuous control.
For CarRacing, small random shifts may reduce overfitting to exact camera
alignment and improve perception of curves/edges.

**Implementation**

This is not native SB3 PPO. Options:

- Add a train-only observation wrapper with small random translation.
- Implement augmentation inside a custom policy/features extractor.
- More invasive: customize PPO minibatch training to augment observations in
  the rollout buffer.

Start conservative:

- Random shift by 2-4 pixels.
- No color jitter if using grayscale.
- No large random crop that removes lookahead.

**Expected effect**

- Mean: uncertain.
- Variance: possible reduction through robustness.

**Cost/risk**

Medium to high. Easy to accidentally make observations partially inconsistent
between train/eval.

**Multi-car relevance**

Good. Robust visual features should help when other cars appear in view.

### 12. Beta-Distribution Or Tanh-Squashed PPO Policy

**What it does**

Replaces clipped Gaussian continuous actions with a distribution that respects
bounded action ranges.

**Why it should help**

SB3 notes that Gaussian policies are clipped for PPO/A2C and that squashing or
Beta distributions can handle action bounds more correctly. A Beta-PPO paper
reported improved CarRacing success rate with bounded continuous actions.

**Implementation**

This is not an easy native SB3 switch.

Options:

- Implement a custom SB3 distribution and actor-critic policy.
- Use an external PPO implementation with Beta policy support.
- Treat as a research fork after the simpler SB3-compatible experiments plateau.

**Expected effect**

- Mean: potentially high.
- Variance: potentially lower due to less action clipping bias.

**Cost/risk**

High. It may interact poorly with the current gSDE recipe and complicate
checkpoint compatibility.

**Multi-car relevance**

Good conceptually, but not the next step. Keep the current stack simple until
single-car feedforward PPO is closer to its ceiling.

## Lower-Priority Or Defer

### Crop Bottom Indicator Bar

Do not prioritize this. The bottom bar exposes speed, ABS, steering, and gyro
state, all of which are helpful for braking and stability. If visual clutter is
a concern, test masking only reward digits first, not the full indicator bar.

### Color Instead Of Grayscale

Color could help detect road/grass and red-white turn borders, but it increases
input size. Try only after the control/reward experiments above. If tested,
compare:

- grayscale `96x96x4`
- RGB `96x96x4`
- RGB `64x64x2` with RL-Zoo-style skip/stack

### Lap Completion Bonus

Mostly redundant with the built-in tile reward and termination. It may improve
credit assignment, but it is not targeted at sharp-turn failures and can distort
raw reward comparisons.

### Full Centerline Reward

Lower priority than edge-margin penalty. Full centerline shaping can produce
safe-but-slow behavior and may transfer poorly to overtaking.

### RecurrentPPO/LSTM

Still a strong eventual candidate for closing the last gap to 880-900, but it
complicates the multi-car/self-play path. Keep deferred unless feedforward PPO
plateaus after the lower-cost work.

## Suggested Immediate Run Plan

### Run 0: Measurement Baseline

No training change.

- Evaluate `checkpoints/best_740_recipe_v2.zip` for 50 episodes.
- Evaluate `checkpoints/ppo_carracing_final.zip` for 50 episodes.
- Save per-episode reward, seed, length, and lap-finished flag.

Decision rule:

- If final has better p10/failure rate than best, revisit checkpoint selection
  before training anything else.

### Run 1: Stability Guardrail

Change only:

- `target_kl=0.03`
- robust eval/checkpointing

Decision rule:

- Keep if p10 improves or failure count drops without hurting mean by more than
  about 25 points.

### Run 2: Hard-Turn Fine-Tune

Change only:

- failure-seed/hard-turn sampling from the 740 model

Decision rule:

- Keep if random-eval failure count drops and hard-seed eval improves.
- Reject if normal random p10 falls, even if failure-seed scores improve.

### Run 3: Edge-Margin Penalty

Change only:

- near-edge penalty with small `k_edge`

Decision rule:

- Keep if failures drop without visibly making the policy hug the centerline
  everywhere.

### Run 4: Steering-Rate Penalty

Change only:

- `k_delta_steer=0.01` or `0.02`

Decision rule:

- Keep if oscillation and off-track tail improve without slower turn-in.

### Run 5: RL-Zoo Deltas

Change one preprocessing/hyperparameter delta at a time:

1. `batch_size=128`
2. `64x64` resize
3. `FrameSkip(2)` with `frame_stack=2`

Decision rule:

- Keep any delta that improves p10/failure count at equal or better mean.
- Treat frame-skip cautiously if late braking worsens.

## Metrics To Add Before More Long Runs

Track these in evaluation and optionally TensorBoard:

- Episode reward.
- Episode length.
- `lap_finished`.
- Number of grass wheels per step.
- Consecutive grass steps.
- Minimum and average lateral margin.
- Steering delta mean/max.
- Brake usage before high-curvature sections.
- Seed and track identifier.

Recommended summary table per model:

| Metric | Why |
|---|---|
| Mean reward | Overall quality |
| Std reward | Current headline weakness |
| Min reward | Catastrophic tail |
| p10 reward | Reliability under bad tracks |
| Failure count below 500 | Human-readable stability |
| Lap-finish rate | Separates slow success from crash |
| Mean episode length | Detects speed changes |
| Grass-step fraction | Direct off-track signal |

## Final Recommendation

The highest impact-per-effort path is:

1. Improve evaluation/checkpoint selection.
2. Add `target_kl`.
3. Fine-tune on failure/hard-turn seeds.
4. Add a small edge-margin penalty.
5. Add a small steering-rate penalty.
6. A/B the remaining RL-Zoo preprocessing deltas.

This order attacks the observed high-variance failure tail while preserving the
current stable feedforward PPO foundation for the eventual multi-car phase.
