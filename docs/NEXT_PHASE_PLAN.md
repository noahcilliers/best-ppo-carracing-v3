# Next Phase Plan: Reducing Variance in the CarRacing PPO Agent

This is the plan we are executing after Phase 1. It synthesizes three independent
research passes (see the companion docs in this folder) into one prioritized,
sequenced roadmap.

- `docs/codex-research.md` — most grounded in our actual setup; 12 ranked experiments.
- `docs/deep-research-report.md` — best novel ideas (action-space redesign, telemetry); strong sourcing.
- `docs/gemini-research.md` — concise; one real gem (velocity-linked grass penalty).

## Where we are

- Phase 1 baseline: feedforward PPO (`CnnPolicy`/NatureCNN). Honest tail-aware
  baseline (Phase A, 50 fixed seeds): **mean 657, std 233, median 711, min 152,
  p10 329, 32% failure-rate (<500)**. (The earlier "740 +/- 191" was a lucky
  20-random-episode read — which is exactly why we built tail-aware eval first.)
  Worst seeds captured for the hard-turn suite: 5, 19, 12, 43, 13.
- The weakness is **variance, not mean**: occasional catastrophic off-track
  episodes on sharp turns. Two observed causes:
  - (a) sees the turn but brakes/reacts too late (control/training gap)
  - (b) does not perceive the turn early enough (field-of-view limit)

## What all three reports agree on (the high-confidence core)

1. This is a **tail/variance problem, not a mean problem.** Corollary: bigger
   models and longer training are NOT the first levers.
2. **Action-smoothness / steering-rate penalty** — unanimous. Attacks the
   oscillation/overcorrection half of the failure. Cheap.
3. **Tail-aware evaluation & checkpoint selection** — a 5-episode eval can save a
   lucky checkpoint at std 191. Measure p10 / min / failure-rate over ~50 episodes
   on fixed seeds; select by `mean - 0.5*std` (or p10), not noisy mean.
4. **`target_kl`** (~0.015-0.03) — near-zero-risk stability guardrail.
5. **Early-termination on sustained grass** — stops wasted experience on spirals.
6. **No raw speed bonus yet** — it amplifies the exact crashes; use progress-based
   shaping, and only after variance is controlled.

## Divergences and our rulings

- **Frameskip / 64x64 / framestack-2 (RL-Zoo deltas):** SKIP frameskip — it
  worsens late braking, our exact failure. Only `batch_size=128` is worth a cheap test.
- **Eval-driven (plateau) LR scheduling (Gemini):** SKIP — our linear-decay-to-0
  is what fixed the regression; plateau-LR risks reintroducing it.
- **Longer training:** not a first move (4M is already the tuned-recipe budget).

## Unique ideas worth stealing

- **Symmetric 2D action wrapper** (deep-research): collapse gas/brake onto one
  symmetric axis (`+`=gas, `-`=brake). Gymnasium's own source notes the action
  space isn't symmetric; SB3 recommends symmetric spaces. Low code, cleaner brake
  decision — targets cause (a).
- **Structured telemetry -> `MultiInputPolicy`** (deep-research): feed
  speed/steering/yaw as an explicit vector instead of forcing the CNN to read the
  HUD. The car can't brake right if it can't cheaply read its own speed.
- **Velocity-linked grass penalty** (Gemini): scale the grass penalty by speed, so
  high-speed corner-cutting is punished hard but low-speed recovery isn't. Upgrades
  our flat `grass_env.py` and avoids the "timid car" failure.

## The plan (bundle the cheap stuff; isolate the architecture)

Protocol: 2M single-seed pilot to reject losers; promote winners to 4M + 3 seeds.
Every experiment is judged on tail metrics vs the baseline, not mean alone.

Rationale for bundling: the cheap, low-risk, same-direction changes (KL guardrail
+ reward/termination shaping) all pull toward "reduce the off-track tail," so we
run them together in ONE pilot to save compute. We only go back to strict
one-at-a-time when we touch the policy interface/architecture, where attribution
actually matters.

### Phase A — Measurement first (no training)  [DONE]
Tail-aware `evaluate.py` (mean/std/min/p10/failure-rate over fixed seeds) +
`TailAwareEvalCallback` (saves best by `mean - 0.5*std`). Honest baseline:
**657 mean / 233 std / 711 median / 152 min / 329 p10 / 32% failure-rate** over 50
fixed seeds. Worst seeds (hard-turn suite): 5, 19, 12, 43, 13.

### Phase B — Stability + Control bundle (ONE run)
All four together, each a configurable flag (default off = clean baseline):
- `target_kl ~= 0.03` — KL guardrail against oversized updates.
- action-smoothness penalty (`k_smooth`) — kills steering oscillation.
- velocity-linked grass penalty (`k_grass_speed`) — punishes high-speed
  corner-cutting, not low-speed recovery (avoids a timid car).
- early-termination on sustained grass — ends unrecoverable off-track spirals.
Pilot 2M vs the 657 baseline; if failure-rate / p10 improve, confirm at 4M.
If results are mixed, THEN bisect to find the culprit.

### Phase C — Architecture (strict one-at-a-time)
1. **Symmetric 2D action wrapper** — cleaner brake decision. *(cause a)*
2. **Structured telemetry (`MultiInputPolicy`)** — explicit speed/steering/yaw so
   the policy can brake correctly. *(cause a)*

### Phase D — Perception (the "doesn't see" half)
The **field-of-view widening** experiment, best bundled with the speed phase
(a faster car genuinely needs more lookahead). *(cause b)*

### Phase E — Only after variance is controlled
Progress-based (not raw) speed shaping -> then research-tier (wider CNN, DrAC
pixel-augmentation, Beta policy) if still short.

## Skip / defer
Raw speed bonus early - frameskip - eval-driven LR - RGB - cropping the HUD
(expose telemetry instead) - longer-training-as-first-move - RecurrentPPO.

## Evaluation metrics we now track (per model)

| Metric | Why |
|---|---|
| Mean reward | Overall quality |
| Std reward | The headline weakness |
| Min reward | Catastrophic tail |
| p10 reward | Reliability on bad tracks |
| Failure-rate (< threshold) | Human-readable stability |
| Mean episode length | Detects speed changes |

## The meta-conclusion

All three reports independently converge: **smooth the control, make the brake
decision easier, and measure the tail — don't scale the model.** That is exactly
aimed at the 740 +/- 191 baseline.
