"""Convert public mouse-dynamics datasets into the recorder's trial format.

All three log raw events as timestamp / button / state / x / y, so one parser handles them:
  sapimouse  https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.html   (120 users, browser, ~8 MB)
  boun       https://data.mendeley.com/datasets/w6cxr8yc7p/2              (24 users, desktop, CC BY 4.0, ~4.9 GB)
  balabit    https://github.com/balabit/Mouse-Dynamics-Challenge          (10 users, remote desktop)

Each movement between two button presses becomes a trial. Drags are dropped, long pauses split off,
and because these logs don't know what was clicked, target width is set to --default-width.

    python scripts/import_public.py sapimouse data/external/sapimouse
    python scripts/train.py --data "data/public/sapimouse/*.json" --out checkpoints/pretrained.pt
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DRAG_PX = 5.0


def find_col(header, *names):
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    for n in names:
        for i, h in enumerate(low):
            if len(n) > 1 and n in h:
                return i
    return None


def event_kind(state):
    s = state.strip().lower()
    if s.startswith("press") or s in ("down", "mousedown"):
        return "press"
    if s.startswith("release") or s in ("up", "mouseup"):
        return "release"
    return "move"


def read_events(path):
    """[(t_ms, x, y, kind)] from a CSV event log, or None if the file isn't one."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        ti = find_col(header, "client timestamp", "timestamp", "time")
        xi, yi, si = find_col(header, "x"), find_col(header, "y"), find_col(header, "state")
        if None in (ti, xi, yi, si):
            return None
        rows = []
        for r in reader:
            try:
                rows.append((float(r[ti]), float(r[xi]), float(r[yi]), event_kind(r[si])))
            except (ValueError, IndexError):
                continue
    if len(rows) < 2:
        return None
    t = np.array([r[0] for r in rows])
    dt = np.diff(t)
    dt = dt[dt > 0]
    to_ms = 1000.0 if len(dt) and np.median(dt) < 0.5 else 1.0  # seconds vs milliseconds
    return [(r[0] * to_ms, r[1], r[2], r[3]) for r in rows]


def make_trial(buf, width, max_pause_ms, max_dwell_ms):
    arr = np.array(buf, dtype=np.float64)
    # A long pause mid-segment means the user stopped; keep only the final movement.
    gaps = np.nonzero(np.diff(arr[:-1, 0]) > max_pause_ms)[0]
    if len(gaps):
        arr = arr[gaps[-1]:].copy()
        arr[0, 0] = arr[1, 0] - 10.0
    if len(arr) < 3:
        return None
    arr[-1, 0] = min(arr[-1, 0], arr[-2, 0] + max_dwell_ms)
    events = arr.tolist()
    return {
        "start": events[0][1:],
        "target_box": None,
        "target_size": [width, width],
        "target_size_estimated": True,
        "events": events,
        "click": events[-1],
        "misses": [],
    }


def segment(events, width, max_pause_ms, max_dwell_ms):
    trials, buf, press_pos = [], [], None
    for t, x, y, kind in events:
        if kind == "press":
            if buf:
                trial = make_trial(buf + [(t, x, y)], width, max_pause_ms, max_dwell_ms)
                if trial:
                    trials.append(trial)
            buf, press_pos = [(t, x, y)], (x, y)
        elif kind == "release":
            if press_pos and np.hypot(x - press_pos[0], y - press_pos[1]) > DRAG_PX:
                buf = [(t, x, y)]  # it was a drag; the next pointing movement starts here
            else:
                buf.append((t, x, y))
            press_pos = None
        elif buf:
            buf.append((t, x, y))
    return trials


def user_key(path, root):
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else path.stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("format", choices=["sapimouse", "boun", "balabit"])
    p.add_argument("root", help="folder containing one subfolder per user")
    p.add_argument("--out", help="default: data/public/<format>")
    p.add_argument("--default-width", type=float, default=24.0)
    p.add_argument("--max-pause-ms", type=float, default=500.0)
    p.add_argument("--max-dwell-ms", type=float, default=300.0)
    p.add_argument("--max-trials-per-user", type=int, default=0, help="0 = no cap; useful for boun")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    while True:  # descend through single wrapper folders (e.g. an unzipped top-level dir)
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(f.is_file() for f in root.iterdir()):
            root = subdirs[0]
        else:
            break

    files = [f for f in sorted(root.rglob("*")) if f.is_file() and f.suffix.lower() in ("", ".csv", ".txt")]
    by_user = defaultdict(list)
    skipped = 0
    for f in files:
        events = read_events(f)
        if events is None:
            skipped += 1
            continue
        by_user[user_key(f, root)].extend(segment(events, args.default_width, args.max_pause_ms, args.max_dwell_ms))

    out = Path(args.out or f"data/public/{args.format}")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    total = 0
    for user, trials in by_user.items():
        if args.max_trials_per_user and len(trials) > args.max_trials_per_user:
            trials = rng.sample(trials, args.max_trials_per_user)
        if not trials:
            continue
        (out / f"{user}.json").write_text(json.dumps({"source": args.format, "user": user, "trials": trials}))
        total += len(trials)
    print(f"{len(files) - skipped} logs, {len(by_user)} users, {total} trials -> {out}  ({skipped} files skipped)")


if __name__ == "__main__":
    main()
