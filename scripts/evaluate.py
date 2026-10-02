"""Compare generated trajectories against held-out real ones.

For each real trial, the model generates a path with the same start, end and
target width. Distributions of kinematic metrics are compared with the KS
statistic (0 = identical distributions, 1 = disjoint).

    python scripts/evaluate.py --ckpt checkpoints/model.pt --data "data/raw/*.json" --plot eval.png
    # also compare paths the Playwright demo actually produced in the browser:
    python scripts/evaluate.py --ckpt checkpoints/model.pt --compare "data/bot/*.json"
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanmouse.data import canonicalize, load_trials, trial_points, trial_width  # noqa: E402
from humanmouse.generate import PathGenerator  # noqa: E402
from humanmouse.metrics import METRICS, fitts_fit, ks_statistic, summarize, trajectory_metrics  # noqa: E402


def collect(trials, step_ms, limit):
    out = []
    for t in trials:
        pts = trial_points(t, step_ms)
        if pts is not None:
            out.append((pts, trial_width(t)))
        if len(out) >= limit:
            break
    return out


def describe(name, paths, step_ms):
    rows = [trajectory_metrics(p, step_ms) for p, _ in paths]
    dists = [np.hypot(*(p[-1] - p[0])) for p, _ in paths]
    a, b, r = fitts_fit([r["duration_ms"] for r in rows], dists, [w for _, w in paths])
    return name, rows, (a, b, r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="trained model (default: the bundled one)")
    p.add_argument("--data", default="data/raw/*.json")
    p.add_argument("--compare", help="glob of recorded bot sessions to include")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", help="save a comparison figure to this path")
    args = p.parse_args()

    gen = PathGenerator(args.ckpt, seed=args.seed)
    trials = load_trials(args.data)
    random.Random(args.seed).shuffle(trials)
    real = collect(trials, gen.step_ms, args.n)
    if not real:
        sys.exit("no usable real trials")
    fake = [(gen.path(pts[0], pts[-1], w, args.temperature)[0], w) for pts, w in real]

    groups = [describe("real", real, gen.step_ms), describe("model", fake, gen.step_ms)]
    if args.compare:
        bot = collect(load_trials(args.compare), gen.step_ms, args.n)
        if bot:
            groups.append(describe("browser", bot, gen.step_ms))

    ref = groups[0][1]
    header = f"{'metric':<18}" + "".join(f"{g[0]:>22}" for g in groups) + "".join(f"{'KS ' + g[0]:>14}" for g in groups[1:])
    print(f"n = {', '.join(f'{g[0]}: {len(g[1])}' for g in groups)}\n")
    print(header)
    for m in METRICS:
        cells = "".join(f"{summarize(g[1])[m][0]:>12.3f} +- {summarize(g[1])[m][1]:<7.3f}" for g in groups)
        ks = "".join(f"{ks_statistic([r[m] for r in ref], [r[m] for r in g[1]]):>14.3f}" for g in groups[1:])
        print(f"{m:<18}{cells}{ks}")
    print("\nFitts' law  MT = a + b*log2(D/W+1)")
    for name, _, (a, b, r) in groups:
        print(f"  {name:<8} a = {a:7.1f} ms   b = {b:6.1f} ms/bit   r = {r:.3f}")

    if args.plot:
        plot(real, fake, gen.step_ms, args.plot)
        print(f"\nfigure -> {args.plot}")


def plot(real, fake, step_ms, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (name, paths) in enumerate([("real", real), ("model", fake)]):
        ax_path, ax_speed = axes[0, col], axes[1, col]
        for pts, w in paths[:25]:
            q, _ = canonicalize(pts, w)
            ax_path.plot(q[:, 0], q[:, 1], lw=0.8, alpha=0.7)
            speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.hypot(*(pts[-1] - pts[0]))
            ax_speed.plot(np.arange(len(speed)) * step_ms, speed, lw=0.8, alpha=0.7)
        ax_path.set_title(f"{name}: canonical paths (start 0,0 → target 1,0)")
        ax_path.set_aspect("equal")
        ax_path.set_xlim(-0.2, 1.3)
        ax_path.set_ylim(-0.4, 0.4)
        ax_speed.set_title(f"{name}: speed profile (distance-normalized)")
        ax_speed.set_xlabel("ms")
        ax_speed.set_xlim(0, 1500)
        ax_speed.set_ylim(0, 0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
