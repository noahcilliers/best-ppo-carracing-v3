"""
Diagnose a training run from its TensorBoard logs.

Prints the training reward curve (rollout/ep_rew_mean) over timesteps, plus the
critic-health signal (explained_variance) and policy noise (std), so we can see
whether the run plateaued, regressed, or collapsed late.

Usage:  .venv/bin/python diagnose.py
"""

import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["rollout/ep_rew_mean", "train/explained_variance", "train/std"]


def load_all():
    series = {t: [] for t in TAGS}
    for path in sorted(glob.glob("runs/**/events.out.tfevents.*", recursive=True)):
        acc = EventAccumulator(path)
        acc.Reload()
        for t in TAGS:
            if t in acc.Tags().get("scalars", []):
                for e in acc.Scalars(t):
                    series[t].append((e.step, e.value))
    for t in TAGS:
        series[t].sort(key=lambda x: x[0])
    return series


def main():
    s = load_all()
    rew = s["rollout/ep_rew_mean"]
    if not rew:
        print("No reward data found in runs/. Did training log to runs/?")
        return

    # Sample ~20 evenly spaced points so the curve is readable.
    n = len(rew)
    idxs = [round(i * (n - 1) / 19) for i in range(min(20, n))]
    print(f"{'step':>10} | {'ep_rew_mean':>11} | {'expl_var':>8} | {'std':>5}")
    print("-" * 46)

    def nearest(series, step):
        if not series:
            return None
        return min(series, key=lambda x: abs(x[0] - step))[1]

    for i in idxs:
        step, r = rew[i]
        ev = nearest(s["train/explained_variance"], step)
        std = nearest(s["train/std"], step)
        ev_s = f"{ev:8.3f}" if ev is not None else "   n/a"
        std_s = f"{std:5.2f}" if std is not None else "  n/a"
        print(f"{step:>10} | {r:>11.1f} | {ev_s} | {std_s}")

    # Headline facts.
    best_step, best_val = max(rew, key=lambda x: x[1])
    last_step, last_val = rew[-1]
    print("-" * 46)
    print(f"PEAK reward {best_val:.1f} at step {best_step:,}")
    print(f"LAST reward {last_val:.1f} at step {last_step:,}")
    if best_val - last_val > 30:
        print(">> Reward REGRESSED after its peak (late-training instability).")
    else:
        print(">> Reward PLATEAUED near its peak (no late regression).")


if __name__ == "__main__":
    main()
