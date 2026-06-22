# PRD: Custom Track Layouts and Indy-Style Oval for MultiCarRacing

## Problem Statement

The current multi-car racing environment can place several agents on a shared
CarRacing-style track, but the track shape is generated randomly on reset. That
is useful for general driving, but it does not let us intentionally stage a race
on a recognizable course shape, compare agents across the same map, or train
against a stable racing layout.

The user wants to create a custom map for agents to run on, starting with an
Indy 500 inspired oval. From the user's perspective, this should feel like
choosing a named track layout, then watching multiple cars run laps on that
track with the same observation, reward, collision, and rendering pipeline that
already exists.

## Solution

Add a custom track layout system to the multi-car environment. The first custom
layout will be an Indianapolis-style oval with two long straights, two short
chutes, and four quarter-turn corners. The layout will be generated as track
geometry, not as a static background image, so the environment can build road
tiles, spawn cars, detect progress, assign rewards, render per-car observations,
and support collisions exactly as it does for the existing random tracks.

The user-facing result should be a new environment option such as
`track_layout="indy_oval"` while preserving the current random track behavior as
the default. A user should be able to run multiple agents on the custom oval,
optionally widen the road, enable collisions, and render the race without
changing the policy inference harness in a major way.

## User Stories

1. As an RL practitioner, I want to choose a named track layout, so that I can
   run repeatable races instead of relying only on random tracks.
2. As an RL practitioner, I want the existing random track behavior to remain
   the default, so that current scripts and training workflows do not change.
3. As an RL practitioner, I want an Indy-style oval layout, so that I can stage
   races that resemble the shape and rhythm of the Indy 500.
4. As an RL practitioner, I want the custom oval to be generated from geometry,
   so that road tiles, rewards, and car physics still work normally.
5. As an RL practitioner, I want the oval to include long straights, short
   chutes, and four distinct turns, so that it feels closer to Indianapolis than
   a simple two-arc oval.
6. As an RL practitioner, I want the layout to be deterministic, so that two
   runs with the same configuration produce the same track.
7. As an RL practitioner, I want the layout to support seeded reproducibility,
   so that experiment results can be compared fairly.
8. As an RL practitioner, I want the track to close cleanly, so that cars can
   complete laps without hitting gaps or discontinuities.
9. As an RL practitioner, I want the generated centerline points to be evenly
   spaced, so that reward tiles are consistent around the whole lap.
10. As an RL practitioner, I want the road heading to be computed correctly at
    every track point, so that cars spawn facing the right direction and camera
    views stay stable.
11. As an RL practitioner, I want the road width to remain configurable, so that
    I can give multiple cars room to race side by side.
12. As an RL practitioner, I want the oval to support a wider default racing
    surface than the random track, so that multi-agent racing is less cramped.
13. As an RL practitioner, I want the custom track to work with one car, so that
    I can validate the layout before adding opponents.
14. As an RL practitioner, I want the custom track to work with multiple cars,
    so that agents can race on the shared map.
15. As an RL practitioner, I want cars to spawn on the front straight, so that
    races begin in a recognizable grid position.
16. As an RL practitioner, I want optional multi-column grid starts, so that an
    Indy-style three-wide start can be represented when the road is wide enough.
17. As an RL practitioner, I want spawn rows to be staggered backward from the
    start line, so that cars do not overlap at reset.
18. As an RL practitioner, I want spawn positions to stay inside the road
    boundaries, so that agents begin in valid driving positions.
19. As an RL practitioner, I want spawn assignment to remain optionally
    randomized, so that no car always receives the same starting advantage.
20. As an RL practitioner, I want the existing per-car observations to work on
    the custom track, so that trained policies can be evaluated without a new
    observation format.
21. As an RL practitioner, I want the existing tile-visit reward to work on the
    custom track, so that progress around the oval is rewarded immediately.
22. As an RL practitioner, I want lap completion to work on the custom track, so
    that finishing a lap is detected consistently.
23. As an RL practitioner, I want per-car reward accounting to remain
    independent, so that each agent can be scored fairly.
24. As an RL practitioner, I want the existing collision option to work on the
    oval, so that cars can physically interact in a race setting.
25. As an RL practitioner, I want the track to render in human mode, so that I
    can watch the cars race visually.
26. As an RL practitioner, I want the track to render in state-pixel mode, so
    that policy inference sees the same kind of image it already expects.
27. As an RL practitioner, I want the infield and outer track area to remain
    visually distinct, so that the route is readable in rendered views.
28. As an RL practitioner, I want borders or curbs to appear around the corners,
    so that the custom track is visually easier to interpret.
29. As an RL practitioner, I want the layout to support clockwise or
    counterclockwise direction as a configuration option, so that experiments
    can vary direction if needed.
30. As an RL practitioner, I want the environment info payload to identify the
    selected layout, so that logs and videos can be traced back to the map.
31. As an RL practitioner, I want command-line scripts to accept the track layout
    option, so that I can run the custom oval without editing source code.
32. As an RL practitioner, I want evaluation runs to report the selected layout,
    so that performance numbers are not confused across random and custom maps.
33. As an RL practitioner, I want training scripts to be able to use the custom
    track, so that agents can eventually specialize on oval racing.
34. As an RL practitioner, I want the custom track system to be extensible, so
    that future maps can be added without rewriting the environment.
35. As an RL practitioner, I want future layouts to reuse the same validation
    checks, so that new maps fail fast when their geometry is invalid.
36. As a developer, I want the track geometry generator isolated from Box2D, so
    that it can be unit-tested without standing up the physics world.
37. As a developer, I want road tile construction shared between random and
    custom layouts, so that both layout types use the same collision and reward
    surface.
38. As a developer, I want the layout interface to return a simple centerline
    representation, so that the environment does not care whether the track came
    from random generation or a named layout.
39. As a developer, I want validation helpers for custom layouts, so that
    closure, spacing, heading, and bounds issues are caught early.
40. As a developer, I want focused tests for the pure geometry pieces, so that
    custom tracks can be changed confidently.
41. As a developer, I want smoke tests for environment reset and step on the
    custom oval, so that integration failures are caught before visual testing.
42. As a developer, I want the first custom layout to be implemented without
    changing the model architecture, so that track work and learning work remain
    separate.

## Implementation Decisions

**Overall strategy.** Add named track layouts to the existing environment rather
than creating a separate racing environment. The current multi-car environment
already has the hard pieces: per-car cameras, per-car rewards, shared physics,
multi-car spawning, optional collisions, and human rendering. A custom layout
should feed that pipeline instead of duplicating it.

**Track layout option.** Add a constructor option that selects the track layout.
The initial values should be the existing random layout and a new Indy-style
oval layout. The random layout remains the default to preserve existing
behavior.

**Track geometry module.** Introduce a deep module responsible for generating
centerline geometry. Its public interface should accept layout parameters and
return a closed sequence of track points in the same semantic format the
environment already consumes: a progress coordinate, a lateral heading, and a
centerline position. This module should not import Box2D, pygame, or the
environment class.

**Oval construction.** Build the Indy-style oval from geometric primitives:
straight segments and circular arcs. The shape should have two long straights,
two short chutes, and four quarter-turn corners. The generator should resample
the full loop at a consistent detail step so reward tile density remains stable
across straights and corners.

**Approximation policy.** The initial track should be "Indy-style," not a
claim of exact Indianapolis Motor Speedway reproduction. Exact real-world
dimensions, banking, pit lane placement, wall geometry, and surface details can
be added later if the user wants higher fidelity.

**Shared road builder.** Separate the current road tile creation from the random
track generation. Both random and custom layouts should pass centerline points
through one road builder that creates sensor tiles, assigns colors, sets tile
indices, tracks per-car visits, and adds visual border polygons.

**Layout validation.** Add validation for generated layouts before creating the
physics bodies. The validator should check that the track has enough points,
closes cleanly, has no zero-length segments, has usable headings, stays inside
the playfield, and produces road tile polygons with positive area.

**Spawn strategy.** Keep the existing spawn behavior for the random track, but
allow layout-aware spawn metadata for custom tracks. The Indy-style oval should
define the start line on the front straight and support configurable grid
columns. The default can remain two columns for compatibility, with an option
for a three-wide Indy-style start when the track is wide enough.

**Track width.** Preserve the existing road width scaling option. For the
Indy-style oval, document or default to a wider road setting during demos so
multiple cars can run side by side without starting in an overly cramped state.

**Direction.** The layout system should support a direction option when
practical. Direction affects point order, headings, spawn angles, lap progress,
and border placement, so it should live in the geometry layer rather than being
patched into rendering.

**Rendering.** Reuse the existing renderer. The custom track should produce the
same road polygons and colors expected by the renderer. Visual differences such
as cleaner curbs, infield coloring, or wall-like borders can be layered on top
after the geometry is reliable.

**Reward and lap completion.** Reuse the existing tile-visit reward and lap
completion logic. The custom track should be represented as a full loop of road
tiles so that each car earns progress reward and finishes a lap through the same
mechanism as the random track.

**Command-line integration.** Add track layout arguments to the scripts used for
watching or driving multiple cars. This keeps the first user workflow simple:
choose the layout, number of agents, width scale, collisions, and render mode
from the command line.

**Training implications.** The current single-car policy was trained on random
CarRacing tracks. It may drive on the oval well enough for demos, but
oval-specific racing quality should be treated as a follow-up training problem.
The PRD covers the environment feature, not a guarantee that the current policy
will learn passing, drafting, or optimal oval lines without retraining.

**Extensibility.** Design the custom track system so additional layouts can be
registered by name. Future layouts should be data-light functions that generate
centerline geometry, not hard-coded environment branches scattered through the
physics and render code.

## Testing Decisions

**What makes a good test here.** Tests should assert external behavior of the
layout system: generated tracks are closed, well-spaced, bounded, consistently
oriented, spawnable, and compatible with environment reset and step. Tests
should not lock onto private helper details or exact floating-point coordinates
unless those values are part of an explicit public contract.

**Pure unit tests for track geometry.** The geometry module should be tested
without Box2D or pygame. Tests should cover the Indy-style oval point count,
closed-loop distance, segment spacing tolerance, heading consistency, clockwise
or counterclockwise direction, configurable dimensions, and deterministic output
for the same parameters.

**Unit tests for layout validation.** Validation should reject empty tracks,
tracks with too few points, zero-length segments, non-finite coordinates,
non-closing loops, invalid headings, and tracks that exceed the playfield. These
tests should use small synthetic inputs so failures are easy to understand.

**Unit tests for spawn behavior.** Spawn tests should verify that cars start
inside the road width, rows are staggered behind the start line, multi-column
starts do not overlap, a single car remains centered, and randomized assignment
changes car-to-slot mapping without changing the set of valid slots.

**Integration smoke tests.** Add at least one smoke test that creates the
environment with the Indy-style oval, resets it, takes a few no-op or low-speed
actions, and verifies observation shape, reward shape, termination payloads, and
absence of exceptions. This test may be skipped when Box2D dependencies are not
installed, matching the style of environment-dependent checks.

**Rendering checks.** The first implementation should include manual or
scripted visual verification in human or RGB-array mode. Automated pixel-perfect
rendering tests are not recommended initially because antialiasing and pygame
surface behavior can be brittle across platforms.

**Prior art.** Follow the existing pattern of isolating pure helper logic behind
small modules with focused tests. The current observation preprocessor tests are
a good model: synthetic inputs, public-interface assertions, no dependence on a
display, and no reliance on a trained checkpoint.

## Out of Scope

- Exact recreation of Indianapolis Motor Speedway dimensions or branding.
- Real IndyCar physics, tire models, aerodynamic drafting, fuel strategy, or
  race-control rules.
- Pit lane behavior, pit stops, yellow flags, pace laps, or rolling restarts.
- Physical retaining walls as colliding track barriers in the first version.
- A new model architecture or new observation format.
- Self-play retraining for oval racecraft.
- Multi-agent training library integration beyond keeping the environment
  compatible with the existing array-based action and observation flow.
- Procedural track editors or a visual map-authoring UI.
- Importing track layouts from external CAD, GIS, SVG, or image files.

## Further Notes

The highest-leverage design choice is to treat a track as geometry first. If the
Indy-style oval can produce the same kind of centerline and road tiles as the
random generator, the rest of the system gets to stay boring: cars spawn, cameras
render, tiles pay reward, collisions happen, and laps finish through the same
contracts already used by the current environment.

The first milestone should be a visually correct oval with one car. The second
milestone should be multiple cars spawned cleanly on the front straight. The
third milestone should be policy-driven multi-car demos with collisions and a
wider road. Only after those are stable should the project move into
oval-specific reward shaping or retraining.

Because an oval has long straights and repeated left turns, it may expose policy
weaknesses differently than the random tracks. That is useful: a fixed custom
track gives us a controlled benchmark for comparing agents, rewards, collision
settings, and future self-play training.
