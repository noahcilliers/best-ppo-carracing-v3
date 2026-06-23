# best-ppo-carracing-v3

A reinforcement-learning sandbox for training and inspecting CarRacing agents.
The project starts with a strong single-car `CarRacing-v3` PPO driver, then uses
that policy as the base for a modern multi-car racing environment with shared
tracks, per-car observations, optional collisions, and named custom layouts.

## What is in here

- Stable-Baselines3 PPO training for Gymnasium `CarRacing-v3`.
- Grayscale, 4-frame-stacked image observations for motion-aware control.
- Optional reward shaping for grass/off-track behavior, action smoothness, and
  sustained off-track termination.
- Tail-aware checkpoint evaluation over fixed seeds, selecting for reliability
  instead of a lucky mean reward.
- A Gymnasium-compatible `MultiCarRacing` environment with N cars in one Box2D
  world.
- A deterministic Indy-style oval layout, configurable start grids, overview
  rendering, and scripted map demos.

## Current status

The best recorded solo policy is the Phase B model from the project notes:

| Metric, 50 fixed seeds | Phase B |
| --- | ---: |
| Mean reward | 868 |
| Std dev | 84 |
| Minimum reward | 535 |
| P10 reward | 800 |
| Failures below 500 | 0/50 |

Model checkpoints and TensorBoard logs are intentionally ignored by git because
they can be large. If you clone this repo fresh, train a new checkpoint or copy
an existing one into `checkpoints/`.

## Setup

This repo has been developed with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "gymnasium[box2d]" "stable-baselines3[extra]" torch tensorboard huggingface-sb3 pytest
```

If Box2D installation fails, install SWIG first, then rerun the dependency
install:

```bash
python -m pip install swig
python -m pip install "gymnasium[box2d]"
```

## Test

```bash
.venv/bin/python -m pytest
```

The tests cover reward shaping, observation preprocessing, spawn geometry, custom
track generation, and the Indy-style multi-car environment path.

## Train a solo driver

Quick smoke run:

```bash
.venv/bin/python train.py --timesteps 20000 --n-envs 8
```

Longer baseline run:

```bash
.venv/bin/python train.py --timesteps 2000000 --n-envs 8
```

Reward-shaped stability run:

```bash
.venv/bin/python train.py \
  --timesteps 4000000 \
  --n-envs 8 \
  --k-grass-speed 0.05 \
  --k-smooth 0.01 \
  --grass-terminate-steps 50 \
  --grass-terminate-penalty 25
```

Training writes checkpoints under `checkpoints/` and TensorBoard logs under
`runs/`.

```bash
.venv/bin/tensorboard --logdir runs/
```

Use `--resume checkpoints/path.zip` to continue the same run after interruption.
Use `--warm-start checkpoints/path.zip` when starting a new objective from
existing weights, such as fine-tuning with changed reward shaping.

## Evaluate a checkpoint

```bash
.venv/bin/python evaluate.py \
  --model checkpoints/best/best_model.zip \
  --episodes 50 \
  --dump docs/eval_run.json
```

Evaluation reports mean, standard deviation, minimum reward, p10 reward, median,
tail-aware score, and failure rate over fixed seeds.

## Watch a trained policy

```bash
.venv/bin/python watch_model.py --model checkpoints/best/best_model.zip
```

To compare against a pretrained Hugging Face SB3 policy:

```bash
.venv/bin/python watch_pretrained.py
```

## Inspect multi-car racing

Show the custom Indy-style oval with scripted cars:

```bash
.venv/bin/python map_view.py \
  --track-layout indy_oval \
  --num-agents 6 \
  --spawn-columns 3 \
  --track-width-scale 2.0
```

Save a headless overview image:

```bash
.venv/bin/python map_view.py \
  --headless \
  --track-layout indy_oval \
  --num-agents 6 \
  --spawn-columns 3 \
  --track-width-scale 2.0 \
  --save /tmp/indy_overview.png
```

Run one trained single-car policy for every car in the multi-car environment:

```bash
.venv/bin/python drive_multi.py \
  --model checkpoints/best/best_model.zip \
  --num-agents 6 \
  --track-layout indy_oval \
  --spawn-columns 3 \
  --track-width-scale 2.0 \
  --overview
```

Try the scripted collision demo:

```bash
.venv/bin/python collision_demo.py \
  --track-layout indy_oval \
  --track-width-scale 2.0 \
  --spawn-columns 3 \
  --overview
```

## Repository map

| Path | Purpose |
| --- | --- |
| `train.py` | PPO training entrypoint for single-car `CarRacing-v3` |
| `grass_env.py` | Reward-shaping wrapper and CarRacing factory |
| `callbacks.py` | Tail-aware evaluation callback for checkpoint selection |
| `evaluate.py` | Fixed-seed checkpoint evaluation CLI |
| `watch_model.py` | Render a local trained checkpoint |
| `watch_pretrained.py` | Load and render a pretrained Hugging Face SB3 model |
| `multicar/` | Modern Gymnasium multi-car racing environment and helpers |
| `map_view.py` | Scripted full-map custom-layout viewer |
| `drive_multi.py` | Run a trained policy across multiple cars |
| `collision_demo.py` | Scripted car-to-car collision demo |
| `tests/` | Unit and integration tests |
| `docs/` | Project records, PRDs, research notes, and evaluation snapshots |

## Notes for contributors

- `checkpoints/`, `runs/`, `.venv/`, caches, and local agent instructions are
  ignored.
- The committed docs contain the project history and experiment rationale. Start
  with `docs/PROJECT_RECORD.md` for the current roadmap.
- The multi-car environment is usable for inspection and pipeline validation,
  but the shared-policy multi-agent training wrapper is still future work.
