# Custom Track and Map View Overview

## Purpose

This work adds a first custom racing layout to the multi-car environment: an
Indy-style oval. The goal is to move beyond random CarRacing tracks so agents can
run, race, and eventually learn on a stable map that is easy to inspect.

The custom track is implemented as geometry, not as a background image. That
means the existing environment still owns the important simulation behavior:
road tiles, per-car rewards, lap progress, collisions, egocentric observations,
and pygame rendering.

## What Changed

### Custom Track Layouts

`MultiCarRacing` now accepts named layouts:

- `track_layout="random"` keeps the existing random CarRacing-style track.
- `track_layout="indy_oval"` builds a deterministic oval with long straights,
  short chutes, and four quarter-turn corners.

The custom oval is generated in `multicar.tracks`, which is pure geometry code.
It has no Box2D or pygame dependency, so it can be tested independently of the
simulation.

### Three-Wide Starts

Spawn placement now supports configurable grid columns:

- `spawn_columns=2` preserves the original two-wide behavior.
- `spawn_columns=3` creates a wider Indy-style starting grid.

For the oval demos, `track_width_scale=2.0` is useful because it gives multiple
cars enough room to start side by side.

### Full-Map Overview Rendering

The environment now has full-track spectator rendering in addition to the
existing per-car camera views:

- `render_mode="human"` shows tiled egocentric car cameras.
- `render_mode="human_overview"` opens a whole-map spectator window.
- `render_mode="rgb_array_overview"` returns a headless full-map RGB frame.

The overview camera fits the whole road into the window and draws colored car
markers so the agents remain visible when zoomed out.

### Standalone Map Viewer

`map_view.py` gives a quick way to inspect the custom layout without loading a
trained PPO checkpoint. By default it uses a simple scripted cruise controller so
the cars visibly move around the oval.

Use `--static` when you want the cars parked on the start grid.

## How To Run It

### Moving Full-Map Demo

```bash
.venv/bin/python map_view.py --track-layout indy_oval --num-agents 6 --spawn-columns 3 --track-width-scale 2.0
```

This opens the whole-map spectator view and uses scripted controls to move the
cars around the oval.

### Static Start Grid

```bash
.venv/bin/python map_view.py --track-layout indy_oval --num-agents 6 --spawn-columns 3 --track-width-scale 2.0 --static
```

### Save A Map Preview

```bash
.venv/bin/python map_view.py --headless --track-layout indy_oval --num-agents 6 --spawn-columns 3 --track-width-scale 2.0 --save /tmp/indy_overview.png
```

This is useful if the live pygame window opens behind other macOS windows.

### Run The Existing Policy On The Oval

```bash
.venv/bin/python drive_multi.py --num-agents 6 --track-layout indy_oval --spawn-columns 3 --track-width-scale 2.0 --overview
```

The policy still receives normal egocentric `96x96` observations. The overview
is only a spectator/debug view.

### Scripted Collision Demo On The Oval

```bash
.venv/bin/python collision_demo.py --track-layout indy_oval --track-width-scale 2.0 --spawn-columns 3 --overview
```

## Implementation Map

| File | Role |
|------|------|
| `multicar/tracks.py` | Generates and validates named track geometry |
| `multicar/multi_car_racing.py` | Selects layouts, builds road tiles, renders overview views |
| `multicar/spawn.py` | Computes two-wide or three-wide grid starts |
| `map_view.py` | Standalone moving/static whole-map viewer |
| `drive_multi.py` | Runs the trained policy with custom track and overview options |
| `collision_demo.py` | Scripted collision demo with custom track and overview options |
| `tests/test_tracks.py` | Tests pure track geometry and validation |
| `tests/test_spawn.py` | Tests start-grid placement |
| `tests/test_indy_env.py` | Tests env reset/step, overview render, and scripted cruise movement |

## Behavior Notes

- The oval is "Indy-style," not an exact Indianapolis Motor Speedway replica.
- The cars still use Gymnasium's Box2D CarRacing physics.
- `map_view.py` movement is scripted and visual only; it is not a learned racing
  policy.
- The trained policy may not drive the oval well yet because it was trained on
  random CarRacing tracks.
- Full-map overview rendering does not change the agent observation pipeline.

## Verification

The current implementation has tests covering:

- deterministic oval generation
- track closure and spacing
- clockwise/counterclockwise heading behavior
- two-wide and three-wide spawn geometry
- environment reset/step on the custom oval
- full-map overview RGB rendering
- scripted cruise actions moving cars in the map viewer

The latest full test run passed with:

```text
23 passed
```

## Next Useful Steps

1. Tune the scripted map-view controller so cars complete cleaner laps at higher
   speed.
2. Add a lap-time or progress-per-second reward for oval-specific learning.
3. Train or fine-tune a policy on `track_layout="indy_oval"`.
4. Add track boundary walls if we want crashes to be physically constrained by
   the oval.
5. Add additional named layouts once the layout interface settles.
