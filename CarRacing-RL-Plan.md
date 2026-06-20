# CarRacing RL Agent — Build Plan

**Goal:** Train a model to control a car in simulation. Phase 1: keep the car on the track and drive. Phase 2: reward faster lap times / racing.
**Hardware:** Mac Mini (Apple Silicon), 24 GB RAM, 1 TB storage.
**Date:** June 2026

---

## 1. Architecture at a glance

The system has three parts, and it's worth being clear about who provides what:

- **The world — Gymnasium `CarRacing-v3`.** Supplies the track (randomly generated each episode), the car physics (Box2D), the observations (a 96×96 RGB image), and a built-in reward signal. Gymnasium gives you the *environment*, not a trained driver.
- **The agent — Stable-Baselines3 (SB3).** Supplies the learning algorithm (PPO) *and* a default network architecture, including a CNN that reads the pixel observations. You do not write a neural network from scratch.
- **The glue — your code.** A short, commented training script: wrap the env, configure PPO, train, checkpoint, evaluate, and (in Phase 2) shape the reward.

The loop in one sentence: the environment shows the agent an image, the agent outputs an action (steer, gas, brake), the environment advances one physics tick and returns a reward, repeat for millions of steps until the policy learns what drives well.

---

## 2. Build steps

**Step 1 — Environment setup.**
Create a Python virtual environment and install `gymnasium[box2d]`, `stable-baselines3`, and `torch`. Box2D may require `swig` first (via Homebrew: `brew install swig`). This is the step most likely to snag on Mac.

**Step 2 — Sanity check.**
Instantiate `CarRacing-v3`, run a few hundred random-action steps, and confirm observations and rewards flow correctly. Confirms the install works and shows what the agent "sees."

**Step 3 — Phase 1 training (stay on the track).**
Hand the env to SB3's PPO with the CNN policy and train. Use a small stack of parallel environments to make use of the CPU. The default CarRacing reward already encourages "stay on track and make progress," so no custom reward is needed yet — this phase just confirms the agent learns to drive.

**Step 4 — Evaluate and watch.**
Save checkpoints periodically, render an episode to watch the car drive, and log reward curves to track progress.

**Step 5 — Phase 2 training (race faster).**
Shape the reward to emphasize speed: bonus for progress-per-timestep, larger time penalty. This pushes the agent from cautious driving toward lap-time optimization. This is the main custom-reward work.

**Step 6 — Iterate.**
Tune learning rate, entropy, and reward weights. This tuning is the bulk of the real effort; the scaffolding above is the easy part.

**Step 7 — Phase 3: extend the environment to multiple cars and race.**
Modify CarRacing so several cars share one track, then load the trained policy into each and race them. The model side is trivial — a policy is just a saved weights file you can clone N times. The work is all on the environment. See Section 6 for the breakdown. Expect a second training round here (self-play) to get true racecraft, since a solo-trained policy has never seen an opponent.

---

## 3. Disk space

CarRacing is lightweight. Storage is a non-issue on a 1 TB drive — the whole project is a few GB at most.

| Item | Approximate size |
|---|---|
| Python + virtual environment (base) | ~150 MB |
| PyTorch (CPU/Apple Silicon build) | ~1–2 GB |
| Gymnasium + Box2D + SB3 + deps | ~300–500 MB |
| Model checkpoints (~5–20 MB each, many saved) | ~200–500 MB |
| TensorBoard logs | ~50–200 MB |
| Recorded evaluation videos (optional) | ~50–300 MB |
| **Total** | **~2–4 GB** |

You will not run out of space. Plan to keep everything in one project folder so checkpoints and logs stay organized.

---

## 4. Training time

These are **estimates**, and the honest caveat is that they depend heavily on which Apple Silicon chip you have (M1 vs M4, base vs Pro) and how many parallel environments you run. CarRacing uses pixel observations and a CNN, which is the compute-heavy part on a machine with no NVIDIA GPU. Treat the ranges as rough planning numbers, not promises — the right move is to run a short benchmark first and extrapolate.

A standard well-trained CarRacing PPO agent uses on the order of **4 million timesteps**. The variable that sets your wall-clock time is throughput (timesteps/second), which on a CPU-bound Mac is realistically in the low hundreds.

| Milestone | ~Timesteps | What you'll see | Rough time on this machine |
|---|---|---|---|
| First signs of learning | ~250 k | Car stops spinning, follows track briefly | ~30–90 min |
| Visibly competent driving (Phase 1 done) | ~1 M | Stays on track most laps, occasional offs | ~2–5 hours |
| Strong, reliable driving | ~3–4 M | Clean laps, good cornering | ~6–15 hours |
| Racing-tuned (Phase 2) | +2–4 M more | Faster, more aggressive lines | +another 4–15 hours |

Practical implications:
- Expect to run training overnight for the full milestones, not in a single sitting.
- Always checkpoint so a run can be stopped and resumed — you do not want to lose 8 hours to a crash.
- Prefer fewer wasted runs: get the pipeline correct on a short run first, then commit to the long one.
- If timelines feel too slow, the two levers are (a) a short benchmark to measure your actual throughput, and (b) renting a cloud GPU box for the long Phase 2 runs while keeping development local.

---

## 5. Reward design notes

- **Phase 1** uses CarRacing's built-in reward: positive for each new track tile visited, small negative per timestep, penalty for leaving the track. No changes needed — the goal is just "learn to drive."
- **Phase 2** modifies the reward to value speed: increase the per-timestep penalty and/or add a bonus proportional to forward progress per tick, so the agent is pushed to complete the track faster rather than crawl safely. Reward shaping is iterative — small weight changes can meaningfully change behavior, so change one thing at a time.

---

## 6. Extending CarRacing to multiple cars (Phase 3)

Decision: extend `CarRacing` to support multiple cars rather than switching to a lidar-based racer. This keeps the pixel-based pipeline you've already built, at the cost of doing the multi-agent engineering yourself. It's a real but bounded project. Here's what it involves.

**a. Multiple cars in one world.** CarRacing builds one `Car` object in a single Box2D physics world. You instantiate several `Car`s in the same world, each with its own position, action, and state. Box2D handles multiple bodies fine; the change is structural in the env code, not a physics limitation.

**b. Per-car observations (the expensive part).** Each car's observation is a 96×96 image zoomed and centered on *that* car. With N cars you render N camera views every step, so step cost scales with the number of cars — this is the main reason multi-car training is slower than solo. Budget for it.

**c. Car-to-car collisions.** The single-car env has no notion of cars hitting each other. You configure Box2D collision fixtures/filters so cars physically interact (and decide what a collision costs in reward — crash penalty, etc.).

**d. Multi-agent API.** A single-agent Gymnasium env takes one action and returns one observation/reward. A multi-car env takes a set of actions and returns a set of observations/rewards. The standard interface for this is **PettingZoo** (the multi-agent sibling of Gymnasium); restructure the env to that shape so SB3-style training loops can drive all cars.

**e. Per-car reward and termination.** Track tiles visited, lap progress, and done-conditions per car instead of globally.

**f. Self-play training round.** Once the env supports N cars, retrain — typically self-play, where the agent races copies of itself — so it learns blocking, overtaking, and reacting to opponents. Cloning the solo policy gives you N competent-but-oblivious drivers; self-play is what turns that into actual racing.

**Honest scope:** items (a), (c), (d), (e) are moderate code changes to the environment; (b) is the performance cost to watch; (f) is a second training investment comparable to Phase 1–2. Doable as a focused project, but it is genuinely more work than the single-car phases — this is the trade for not switching environments.

---

## 7. Risks and notes

- **Box2D install on Mac** is the most common early snag; `swig` via Homebrew usually fixes it.
- **CPU-bound training is slow.** This is the main constraint of the hardware, not RAM or disk. Low parallelism means patience.
- **Reward hacking.** A poorly shaped Phase 2 reward can produce odd behavior (e.g., the car exploiting the reward instead of racing well). Expect to iterate.
- **MPS (Apple GPU) acceleration** exists in PyTorch but for these small CNNs is often not dramatically faster than CPU and can hit compatibility issues — worth a quick test, not worth blocking on.

---

## 8. Recommended next action

Set up the environment in a sandbox, verify `CarRacing-v3` runs end-to-end, and produce a clean, commented PPO training script in this folder — then do a short benchmark run to replace the time estimates above with numbers measured on your actual machine.
