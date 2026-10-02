"""Generate a synthetic dataset in the recorder's format, for smoke-testing the pipeline.

The generator is a hand-built approximation (Fitts' law timing, a curved
minimum-jerk primary submovement, a corrective submovement, dwell and tremor).
A model trained on it only learns this approximation — train on real recordings
for anything you care about.

    python scripts/make_synthetic.py --trials 2000
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

SIZES = [(12, 12), (16, 16), (24, 24), (36, 36), (56, 56), (80, 80), (60, 20), (90, 28), (140, 36), (220, 44)]
VIEW_W, VIEW_H = 1280, 800


def min_jerk(p0, p1, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def synth_trial(rng, start, t0):
    w, h = SIZES[rng.integers(len(SIZES))]
    for _ in range(100):
        bx, by = rng.uniform(20, VIEW_W - w - 20), rng.uniform(40, VIEW_H - h - 20)
        if math.hypot(bx + w / 2 - start[0], by + h / 2 - start[1]) > 60:
            break
    click = np.array([
        np.clip(rng.normal(bx + w / 2, w / 6), bx + 1, bx + w - 1),
        np.clip(rng.normal(by + h / 2, h / 6), by + 1, by + h - 1),
    ])
    v = click - start
    dist = float(np.hypot(*v))
    u = v / dist
    perp = np.array([-u[1], u[0]])
    mt = (120 + 140 * math.log2(dist / min(w, h) + 1)) * rng.lognormal(0, 0.15)

    dt = 8.0  # ~125 Hz mouse
    aim = click + u * rng.normal(-0.03, 0.05) * dist + perp * rng.normal(0, 0.025) * dist
    n1 = max(3, int(mt * rng.uniform(0.7, 0.85) / dt))
    n2 = max(2, int(mt / dt) - n1)
    s = np.linspace(0, 1, n1)[:, None]
    primary = min_jerk(start, aim, n1) + perp * dist * rng.normal(0, 0.06) * np.sin(np.pi * s)
    corrective = min_jerk(aim, click, n2)[1:]
    dwell = np.repeat(click[None], int(rng.uniform(30, 150) / dt), axis=0)
    pts = np.vstack([primary, corrective, dwell]) + rng.normal(0, 0.3, (n1 + n2 - 1 + len(dwell), 2))
    pts = np.round(pts)
    pts[0], pts[-1] = start, np.round(click)

    times = t0 + np.concatenate([[0], np.cumsum(np.maximum(1.0, rng.normal(dt, 0.6, len(pts) - 1)))])
    events = [[float(t), float(x), float(y)] for t, (x, y) in zip(times, pts)]
    return {
        "start": start.tolist(),
        "target_box": [float(bx), float(by), w, h],
        "target_size": [w, h],
        "events": events,
        "click": events[-1],
        "misses": [],
    }, pts[-1], times[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic/synthetic.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pos, t = np.array([VIEW_W / 2, VIEW_H / 2]), 0.0
    trials = []
    for _ in range(args.trials):
        trial, pos, t = synth_trial(rng, pos, t)
        trials.append(trial)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "synthetic", "viewport": [VIEW_W, VIEW_H], "trials": trials}))
    print(f"{len(trials)} trials -> {out}")


if __name__ == "__main__":
    main()
