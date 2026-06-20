# PRD: Port MultiCarRacing onto Gymnasium CarRacing-v3 and host the trained policy in N cars

## Problem Statement

We have a single-car PPO agent that drives `CarRacing-v3` competently (trained on the current
`gymnasium 1.3.0` / `stable-baselines3 2.9.0` stack). We want to move from one car driving alone to
**multiple cars racing on a shared track**. A community repo (`multi_car_racing`,
`MultiCarRacing-v0`) already implements the multi-agent logic we want — per-car spawning, a
competitive tile-visit reward, and per-car egocentric camera views — but it was written against
`gym~=0.17.2` and a pyglet/OpenGL rendering stack that **no longer exists** in our environment. As
shipped, it cannot even be imported: the legacy `gym` package is not installed, and
`gym.envs.classic_control.rendering.Viewer` has been removed from `gymnasium`. So today there is no
way to put our trained car on a track with opponents.

## Solution

Rebuild the multi-car environment on top of the **current** `CarRacing-v3` source so it runs on our
exact stack, and reuse the old repo only as a *specification* of the multi-agent behavior (its
physics-independent logic), not as code we patch. Concretely:

- Fork the installed `gymnasium` `CarRacing` environment (which already solves rendering in pygame
  and uses the modern API) and graft the multi-agent modifications onto it.
- Extract the genuinely reusable, physics-independent logic from the old repo into small, isolated
  modules: observation preprocessing, spawn geometry, and the competitive reward formula.
- Provide an inference harness that loads our existing trained policy once and drives every car with
  it — no retraining required to *run* a race.
- Deliver the work in independently-runnable phases so each step is verifiable before the next:
  1 car on the new stack → trained policy drives 1 car → 2+ cars sharing a track → car-to-car
  collisions → (future) self-play retraining.

From the user's perspective: a single `MultiCarRacing-v0` environment that accepts `(N, 3)` actions,
returns `(N, 96, 96, 3)` observations and `(N,)` rewards, renders all cars' egocentric views, and can
be driven end-to-end by the already-trained checkpoint.

## User Stories

1. As an RL practitioner, I want to instantiate a multi-car racing environment on my current
   `gymnasium`/`stable-baselines3` stack, so that I can run races without downgrading or reinstalling
   a legacy `gym`.
2. As an RL practitioner, I want to create the environment via `gymnasium.make("MultiCarRacing-v0",
   num_agents=N)`, so that I can choose how many cars race with a single argument.
3. As an RL practitioner, I want the environment to default to a sensible number of agents (2), so
   that the common case works with no configuration.
4. As an RL practitioner, I want `env.reset(seed=...)` to return `(obs, info)` and seed
   reproducibly, so that races are deterministic when I fix the seed.
5. As an RL practitioner, I want `env.step(actions)` to return the modern 5-tuple
   `(obs, reward, terminated, truncated, info)`, so that it is compatible with SB3 2.x and modern
   wrappers.
6. As an RL practitioner, I want each car spawned at a distinct, non-overlapping position on the
   start line, so that cars do not begin stacked on top of each other.
7. As an RL practitioner, I want spawn positions to respect the track winding direction (CW/CCW), so
   that all cars face the correct way at the start.
8. As an RL practitioner, I want spawn positions optionally randomized, so that no car has a fixed
   positional advantage across episodes.
9. As an RL practitioner, I want each car to have a distinct color, so that I can tell them apart in
   the rendered views.
10. As an RL practitioner, I want each car's observation to be a `96×96×3` egocentric image centered
    on that car, so that a per-car policy sees the same kind of input it was trained on.
11. As an RL practitioner, I want the stacked observation returned as shape `(num_agents, 96, 96, 3)`,
    so that I can index each car's view directly.
12. As an RL practitioner, I want to pass actions as a `(num_agents, 3)` array of
    `[steer, gas, brake]`, so that I control all cars in one call.
13. As an RL practitioner, I want each car to receive a competitive tile-visit reward (first car to a
    tile gets the full bonus, later cars a damped share), so that the reward encourages racing rather
    than just driving.
14. As an RL practitioner, I want a small per-timestep penalty applied per car, so that cars are
    incentivized to make progress rather than idle.
15. As an RL practitioner, I want rewards returned as a `(num_agents,)` array, so that I can score
    each car independently.
16. As an RL practitioner, I want per-car termination flags exposed in `info`, so that I can stop
    feeding observations to cars that have finished or left the track.
17. As an RL practitioner, I want the episode to truncate at the registered step limit, so that races
    cannot run forever.
18. As an RL practitioner, I want to load my existing trained checkpoint unchanged and have it drive a
    car, so that I do not have to retrain just to see multi-car driving.
19. As an RL practitioner, I want the inference harness to preprocess each car's frame identically to
    training (grayscale, 4-frame stack, channel-first transpose, `uint8`), so that the policy behaves
    as it did in single-car driving.
20. As an RL practitioner, I want one shared policy to drive all cars via a single batched
    `predict((N,4,96,96))` call, so that inference is efficient and simple.
21. As an RL practitioner, I want a human-viewable render of all cars, so that I can watch a race
    visually.
22. As an RL practitioner, I want human rendering to use a single tiled window rather than one window
    per car, so that it works within SDL2's single-window-per-process limitation on macOS.
23. As an RL practitioner, I want an `rgb_array` render mode, so that I can record race videos.
24. As an RL practitioner, I want a headless render path (no display window), so that I can run/train
    on a machine without a screen.
25. As an RL practitioner, I want the environment to pass SB3's `check_env`, so that I have
    confidence the API contract is correct before training.
26. As an RL practitioner, I want `MultiCarRacing-v0` to reduce exactly to single-car CarRacing when
    `num_agents=1`, so that I can validate the port against known-good behavior.
27. As an RL practitioner, I want the environment registered under `gymnasium`'s registration API, so
    that `gymnasium.make` discovers it.
28. As an RL practitioner, I want cars to physically collide with one another once collisions are
    enabled, so that races involve real contact (blocking, bumping).
29. As an RL practitioner, I want track tiles to remain non-colliding sensors even after car-to-car
    collisions are enabled, so that the friction/reward mechanism is unaffected.
30. As an RL practitioner, I want the work delivered in independently-runnable phases, so that I can
    verify each milestone before investing in the next.
31. As an RL practitioner, I want to (eventually) retrain via self-play, so that cars learn racecraft
    against opponents rather than driving as if alone.
32. As a developer, I want the observation-preprocessing logic isolated in its own module with unit
    tests, so that the highest-risk, silently-failing step is verified independently of the slow
    Box2D/pygame integration.
33. As a developer, I want the spawn geometry isolated as a pure function, so that I can reason about
    car placement without standing up a physics world.
34. As a developer, I want the competitive reward formula isolated as a pure function, so that the
    scoring rule is auditable separately from the Box2D contact callback that feeds it.
35. As a developer, I want the old repo's dependency pins and registration updated to the current
    stack, so that the package installs and imports cleanly.

## Implementation Decisions

**Overall strategy.** Fork the current `gymnasium` `CarRacing` environment and re-apply the old
repo's multi-agent modifications onto it, rather than patching the old file. The old file's rendering
subsystem targets a removed stack (`gym.envs.classic_control.rendering`, pyglet, `pyglet.gl`) and its
`Car.draw` call uses the old 2-argument pyglet signature; the current `Car.draw` takes a pygame
surface plus an explicit camera transform. Patching in place would mean rewriting the entire render
layer from scratch *and* fighting an obsolete API shell, so we start from the side that already runs.

**Modules to build.** Three deep modules with simple, stable interfaces, extracted so they can be
reasoned about (and, for one of them, tested) in isolation:

- **Observation preprocessor.** Interface: takes per-car raw RGB frames `(N, 96, 96, 3) uint8` and a
  rolling state, returns the model-ready batch `(N, 4, 96, 96) uint8` (channel-first, 4 stacked
  grayscale frames). Encapsulates: grayscale conversion using the same luminosity weights as training,
  per-car rolling 4-frame buffers, the cold-start behavior (prime by repeating the first frame ×4),
  the HWC→CHW transpose, and the invariant that pixels stay `uint8` in `[0,255]` with **no** division
  by 255 (SB3's CNN normalizes internally). This is the single most likely source of a silent failure.
- **Spawn geometry.** Interface: takes `num_agents`, the generated track, and the winding direction,
  returns a list of `(x, y, angle)` start poses. Pure math: row/side assignment per car, lateral
  offset along the track normal, longitudinal offset back along the track, and a direction flip for
  CW. No Box2D dependency.
- **Reward accounting.** Interface: given the per-tile per-car visit state and `num_agents` /
  `num_tiles`, computes the per-car reward delta for newly-visited tiles — the first visitor to a tile
  receives the full `1000/num_tiles`, each subsequent visitor a share damped by `1/num_agents`.
  Separated from the Box2D contact callback that detects the visits.

**Integration modules (glue, not deep).** The `MultiCarRacing` `gymnasium.Env` subclass (wires Box2D,
the three modules, and the render loop); the per-car pygame renderer (one egocentric camera transform
per car drawn into a fresh surface); and the inference harness (loads the checkpoint, runs the drive
loop).

**API-surface migration.** Subclass `gymnasium.Env`; `import gymnasium as gym`; import `Car` from the
gymnasium `car_dynamics` module (its constructor signature is unchanged, so the physics port is
clean); `reset(self, *, seed=None, options=None)` calling `super().reset(seed=seed)` and returning
`(obs, info)`; `step` returning the 5-tuple; remove the custom `seed()` method; update metadata keys
to `render_modes` / `render_fps`; move registration to `gymnasium.envs.registration`; pass the full
constructor argument list to `EzPickle.__init__` so the env can be pickled for vectorization.

**Multi-agent interface.** Use the raw `(num_agents, …)` array-return env driven by a hand-written
per-car `predict` loop as the first target — it is a zero-conversion match to the inference harness.
The declared `action_space` and `observation_space` are the **single-car** shapes (`(3,)` and
`(96,96,3)`); the env internally returns stacked `(N,…)` arrays. A PettingZoo `ParallelEnv` adapter is
deferred until a multi-agent training library actually requires it.

**Termination semantics.** Compute env-level scalar `terminated`/`truncated` (OR-reduced across cars;
`truncated` from the registered `max_episode_steps`), and additionally expose per-car termination
flags in `info` so the inference loop can stop feeding finished cars. Off-field exit penalizes and
terminates the offending car.

**Rendering.** Re-implement the per-car egocentric view on pygame: for each car, allocate a fresh
surface, compute the camera transform from that car's hull pose (use `-hull.angle` for heading to keep
observations in the trained policy's distribution), draw the road and *all* cars into the view, then
produce the `96×96` state-pixels array via the env's existing smooth-downsample path. Human mode uses
one tiled `pygame.display` window (cols×rows of car views) updated once per step; headless mode uses
offscreen surfaces only (`SDL_VIDEODRIVER=dummy`, never call `set_mode`).

**Color convention.** Road/grass/border colors are `0–255` ints on the pygame path; car hull colors
stay `0–1` floats because `Car.draw` multiplies by 255 itself. Mixing these up renders all black.

**Collisions (phased).** The old repo has no car-to-car collision filtering and tiles are sensors;
under Box2D defaults, hulls already collide while wheel-wheel contact is filtered. Car-to-car
collisions are deferred to a dedicated phase, where explicit `categoryBits`/`maskBits` are set on hull
and wheel fixtures (tiles stay sensors) and, optionally, a car↔car branch is added to the contact
listener for bump reward/penalty.

**Carry-through details flagged during review.** The current `FrictionDetector` constructor takes
`(env, lap_complete_percent)` and is re-created on `reset` — the per-car rewrite must thread that
argument through or it will raise. Current tiles also carry an index and feed a lap-completion path;
the per-car reward rewrite must either preserve or consciously drop that, since "any car finished"
termination is derived from the per-car tile-visited counts.

**Packaging.** Update `setup.py` to depend on `gymnasium` (drop the `gym~=0.17.2` pin) and update the
package `__init__` to register via `gymnasium.envs.registration`.

## Testing Decisions

**What makes a good test here.** Tests assert *external behavior* of a module through its public
interface, not its internals. For the modules in this PRD that means: given inputs, the returned
arrays have the right shape, dtype, value range, and ordering — never assertions about private buffer
layout or call counts. Tests must not require a Box2D world, a pygame display, or the trained
checkpoint; the modules under test are pure enough to run headless and fast.

**Module that will be unit-tested: the observation preprocessor.** It is the highest-risk component
(its failures are silent — a wandering car, no exception) and the purest (deterministic function of
input frames plus rolling state). Tests should cover:

- Output shape is `(N, 4, 96, 96)` and dtype is `uint8`.
- Values remain in `[0, 255]` and are **not** normalized/divided by 255.
- Grayscale conversion matches the training luminosity weights (assert against a known reference
  conversion, e.g. `gymnasium`'s `GrayscaleObservation`, on a fixed input).
- Channel-first ordering is correct (a frame placed in the stack reappears at the expected channel
  index after transpose).
- Cold-start: on the first frame, all four stacked slots equal that frame.
- Rolling behavior: after K steps the stack holds the last 4 frames in the correct temporal order
  (oldest dropped, newest appended).
- Per-car independence: with `N>1`, each car's buffer evolves independently and frames do not leak
  between cars.

**Prior art.** Mirror the lightweight, dependency-free style of the existing scripts in the project
(e.g. `sanity_check.py`'s direct, assertion-style checks); use fixed synthetic frames (constant or
gradient arrays) as inputs so expected outputs are computable by hand.

**Modules validated by integration/smoke checks rather than unit tests** (per scope decision): spawn
geometry and reward accounting are exercised through the env smoke path and visual inspection; the
env itself is validated by SB3 `check_env`, by the `num_agents=1`-reduces-to-CarRacing equivalence
check, and by watching the trained policy drive. These are explicitly *not* in the unit-test scope of
this PRD.

## Out of Scope

- **Self-play / multi-agent retraining.** Getting cars to learn blocking and overtaking is a separate,
  large training effort; this PRD ends at "the existing policy can drive N cars and they can collide."
- **A PettingZoo `ParallelEnv` interface.** Deferred until a training library requires it; the
  array-return env is the deliverable here.
- **Unit tests for spawn geometry and reward accounting.** Extracted as isolated modules for clarity,
  but only the observation preprocessor is in the unit-test scope of this PRD.
- **Reward shaping for speed / lap-time optimization** beyond the competitive tile reward already
  described.
- **Performance optimization of N-car rendering** beyond the basic measures noted (skipping particle
  draws for state pixels, optional smaller training surface). Large-N throughput tuning is a follow-up.
- **Cloud/GPU training infrastructure.**

## Further Notes

- **Highest-risk item is observation preprocessing, and it fails silently.** Forgetting the HWC→CHW
  transpose, dividing by 255, using mismatched grayscale weights, or feeding fewer than 4 stacked
  frames all yield a CNN running on garbage and a car that merely wanders — with no error. This is why
  it is the one module with unit tests and why Phase 1 validates a single car before scaling to N.
- **Phasing (each phase independently runnable and verifiable).**
  - Phase 0 — fork to a 1-car gymnasium env with all API fixes; verify `make`/`reset`/`step` run and
    `check_env` passes.
  - Phase 1 — wrap the trained policy with the preprocessor around the 1-car env; verify it laps the
    track competently (confirms preprocessing is byte-faithful).
  - Phase 2 — apply the multi-agent deltas (car list, per-car spawn, per-car reward routing, stacked
    returns, per-car render loop); verify two cars spawn offset, lap independently, and both views
    render (cars pass through each other — collisions still off).
  - Phase 3 — enable car-to-car Box2D collisions; verify two cars driven together physically bounce
    and tiles still award reward correctly.
  - Phase 4 (out of scope here) — self-play retrain.
- **Environment context (verified):** Python 3.11.15, `gymnasium 1.3.0`, `stable-baselines3 2.9.0`,
  `Box2D 2.3.10`. The legacy `gym` package is not installed. The saved model has observation space
  `Box(0,255,(4,96,96),uint8)` and action space `Box([-1,0,0],1,(3,),float32)`; the training pipeline
  is `CarRacing-v3` → grayscale (keep dim) → 4-frame stack → channel-first transpose.
- The migration plan and its line-by-line claims were produced by a multi-agent investigation and
  adversarially verified against the actual `gymnasium` and old-repo source files; the five
  highest-risk technical claims (rendering stack removed, `Car` constructor/`draw` signatures, modern
  env API + contact-listener mechanism, absence of collision filtering in the old repo, SB3 requiring
  `gymnasium`) all checked out.
