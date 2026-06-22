# Solo Training — Future Work (shelved)

Phase 1 is complete: the Phase B bundle reached **868 mean / 84 std / 0% failures**
over 50 fixed seeds (`checkpoints/best_868_phaseB_v3.zip`). The variance problem
that motivated further solo work is solved, so these ideas are **shelved**, not
abandoned. Revisit them if we later want to (a) push solo score past ~868 toward
the ~900 ceiling, or (b) improve the base before/after the multi-car phase.

All of these assume the current proven recipe (gSDE, linear LR decay,
`log_std_init=-2`, reward normalization, GELU/`net_arch=[256]`, NO `target_kl`).
Test one at a time, A/B against the 868 baseline on tail metrics (p10 /
failure-rate / min), not mean alone.

---

## Phase C — Architecture (strict one-at-a-time)

### C1. Symmetric 2D action wrapper
- **What:** collapse gas/brake onto one symmetric axis (`+` = gas, `-` = brake),
  exposing `Box(-1, 1, (2,))` instead of the native asymmetric `Box([-1,0,0],[1,1,1])`.
- **Why:** Gymnasium's own source notes the action space isn't symmetric; SB3
  recommends symmetric continuous spaces. Structurally prevents simultaneous
  gas+brake and gives a cleaner braking boundary.
- **How (SB3):** a `gym.ActionWrapper` mapping the 2D action back to the native 3D.
  Keep PPO/gSDE/network unchanged. (Optionally test gSDE `squash_output=True` after.)
- **Effect:** small mean gain, possible variance reduction. **Cost:** low.
- **Multi-car:** cleaner throttle/brake semantics transfer well.

### C2. Structured telemetry (`MultiInputPolicy`)
- **What:** add a `Dict` observation: `{"img": stacked frames, "telemetry":
  [speed, abs0-3, steer_angle, yaw_rate]}`, switch to `MultiInputPolicy`.
- **Why:** the car can't brake correctly if it must read its own speed from a tiny
  HUD strip. Promoting telemetry to an explicit vector directly targets late braking.
- **How (SB3):** wrapper to build the Dict; `PPO("MultiInputPolicy", ...)`; NatureCNN
  for `img`, a small MLP for `telemetry`, concatenate. Normalize telemetry to ~[-1,1];
  keep images at /255 (no obs-normalization).
- **Effect:** medium-high mean gain, large variance reduction if late braking is the
  remaining issue. **Cost:** medium.
- **Multi-car:** very useful — the image gets busier with opponents, ego telemetry
  becomes more valuable.

---

## Phase D — Perception (the "doesn't see the turn" half)

### D1. Widen the camera field-of-view
- **What:** zoom the CarRacing camera out so more upcoming track fits the 96x96 frame.
- **Why:** addresses the "doesn't perceive the turn early enough" failure; matters
  most at high speed.
- **How:** subclass CarRacing and lower its render zoom (env-internals change, not a
  wrapper). Changes the observation -> requires a retrain. Possibly bump resolution
  to keep near-field detail. Tradeoff: wider view = blurrier close-up = can hurt fine
  steering.
- **Effect:** uncertain; unproven beyond the standard view. **Cost:** medium + retrain.
- **Multi-car:** high — lookahead matters more when racing at speed.

---

## Phase E — After variance is controlled (it is)

### E1. Progress-based speed shaping (NOT raw speed bonus)
- **What:** reward forward progress per step (delta arc-length / nearest-tile index),
  optionally a small heading-alignment term; OR slightly increase the time penalty.
- **Why:** pushes faster laps. Use potential-based shaping so it doesn't change the
  optimal policy. A raw speed bonus is avoided — it amplifies sharp-turn crashes.
- **How:** reward wrapper using `env.unwrapped.track` for progress. Start tiny.
- **Effect:** medium-high mean gain if already reliable; risk of re-introducing
  variance if too strong. **Cost:** medium, medium risk (reward hacking).
- **Multi-car:** racing ultimately needs progress-per-time incentives.

### E2. Wider / separate feature extractor
- Modestly wider CNN (e.g. 64/128/128, `features_dim` 512-1024) or
  `share_features_extractor=False`. Medium cost on CPU; try only if perception is the
  bottleneck.

### E3. Pixel augmentation (DrAC-style)
- Random shift/crop with actor-critic-aware regularization (DrAC). External to SB3.
  Improves visual robustness/generalization. Higher cost/risk; do late.

### E4. Beta-distribution policy
- Bounded action distribution (finite support) — reported +63% success on pixel
  CarRacing. Not in SB3 core (custom distribution). High cost; research-tier.

---

## NEW idea — small reward for keeping the same motion as the last step

- **What:** a small POSITIVE reward when the current action (or physical motion)
  matches the previous step — the positive-framed dual of our existing
  action-smoothness penalty. Encourages smooth, continuous control.
- **Why:** we already use a smoothness *penalty* (`-k_smooth * sum|a_t - a_{t-1}|`)
  and it helped. A positive "hold steady" reward is an alternative framing that may
  shape behavior slightly differently and is worth A/B-ing against the penalty.
- **Two interpretations:**
  - **Action continuity (primary):** `reward += k_hold * (1 - mean|a_t - a_{t-1}|)`
    or `reward += k_hold * exp(-c * sum|a_t - a_{t-1}|)` (high when action barely
    changed).
  - **Physical-motion continuity (variant):** reward consistency of heading / speed
    between steps (penalize jerk in the car's actual velocity vector), which targets
    the *result* of smoothness rather than the raw control.
- **How (SB3):** add to `RewardShapingWrapper` in `grass_env.py` as a new knob
  (`k_hold`), store the previous action, default 0.
- **Important nuance / risk:** a positive reward for "sameness" is mathematically
  close to a penalty for "change" (they differ by a roughly constant offset), so the
  gradient effect is similar — but the positive version adds a small per-step bonus
  that can:
  - **reduce urgency** (it partly cancels the `-0.1` time penalty, so the car may
    dawdle / be less inclined to finish), and
  - **reward holding a wrong/frozen action** (e.g., sitting still also "keeps the
    same motion").
  Keep `k_hold` very small, and watch `eval/mean` and lap time so it doesn't make the
  car slow. Best tested as: penalty-only vs reward-only vs both, to see which shaping
  of smoothness drives the cleanest control without timidity.
- **Effect:** small mean change, possible small variance reduction. **Cost:** low.
- **Multi-car:** smooth control helps close-quarters racing and collision avoidance.

---

## Reminder: what NOT to revisit (already ruled out)
`target_kl` (choked from-scratch learning) · raw speed bonus early · frameskip
(hurts late braking) · eval-driven LR scheduling · RGB · cropping the HUD (expose
telemetry instead) · "just train longer" as a first move · RecurrentPPO (deferred to
keep multi-car simpler).
