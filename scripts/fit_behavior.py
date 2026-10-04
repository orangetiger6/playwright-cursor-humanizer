"""Fit drag and scroll timing from recorder sessions.

    python scripts/fit_behavior.py --data "data/raw/*.json" --out checkpoints/behavior.json

Anything with fewer than a handful of recordings keeps its built-in default, so this can be
re-run as more sessions come in.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanmouse.behavior import DEFAULT_MEDIANS, MIN_SAMPLES, fit_behavior, load_tasks  # noqa: E402
from humanmouse.data import load_trials  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/*.json")
    p.add_argument("--out", default="checkpoints/behavior.json")
    args = p.parse_args()

    tasks = load_tasks(args.data)
    fit = fit_behavior(tasks, load_trials(args.data))
    kinds = {k: sum(t["kind"] == k for t in tasks) for k in ("drag", "scroll")}
    print("recorded tasks: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
    print(f"\n{'timing':<18}{'n':>6}{'median ms':>12}{'default':>10}")
    for name, default in DEFAULT_MEDIANS.items():
        v = fit["samples"][name]
        med = f"{np.median(v):.0f}" if v else "-"
        note = "" if len(v) >= MIN_SAMPLES else "  (too few, using default)"
        print(f"{name:<18}{len(v):>6}{med:>12}{default:>10}{note}")
    print(f"\ntrackpad flicks: {len(fit['gestures'])}" + ("" if len(fit["gestures"]) >= MIN_SAMPLES else " (too few, using synthetic)"))
    print(f"drag time scale: {fit['drag_time_scale']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fit, f)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
