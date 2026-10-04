"""Timing and scroll behavior for drags and trackpad scrolling.

These don't need a neural model: a person's press/settle pauses and scroll flicks are
well described by their empirical distributions. `fit_behavior` turns
recorder sessions into a small JSON of samples; `Behavior` draws from it, falling back to
typical values for anything with too few recordings.
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

from .data import resample, trim_onset

MIN_SAMPLES = 8
DEFAULT_BEHAVIOR = Path(__file__).parent / "assets" / "behavior.json"  # the bundled recordings

# Median ms (lognormal, sigma 0.35) used until there are recordings.
DEFAULT_MEDIANS = {
    "drag_press_ms": 150,     # button down -> cursor starts moving
    "drag_settle_ms": 120,    # cursor stops over the drop zone -> button up
    "scroll_pause_ms": 180,   # between flicks while scrolling
}
DEFAULT_DRAG_TIME_SCALE = 1.25  # drags are slower than plain pointing at the same difficulty
SIGMA = 0.35


def load_tasks(pattern):
    """All recorder task records (any kind), tagged with their session's source."""
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            session = json.load(f)
        out.extend(t for t in session["trials"] if "kind" in t)
    return out


def synthetic_gesture(rng, total=None):
    """A trackpad flick: rising finger phase, then exponentially decaying momentum. Returns [[t_ms, dx, dy]]."""
    frame = 16.7
    finger_ms = rng.uniform(80, 200)
    peak = rng.uniform(8, 30)  # px per frame at release
    tau = rng.uniform(220, 380)
    out, t = [], 0.0
    while t < finger_ms:
        out.append([t, 0.0, peak * math.sin(0.5 * math.pi * min(1.0, (t + frame) / finger_ms))])
        t += frame * rng.uniform(0.9, 1.1)
    v, t0 = peak, t
    while v > 0.5:
        v = peak * math.exp(-(t - t0) / tau)
        out.append([t, 0.0, v])
        t += frame * rng.uniform(0.9, 1.1)
    g = np.array(out)
    g[:, 1] = g[:, 2] * rng.normal(0, 0.04)  # slight sideways component
    if total:
        g[:, 1:] *= total / g[:, 2].sum()
    return g


class Behavior:
    def __init__(self, path=None, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.samples, self.gestures = {}, []
        self.drag_time_scale = DEFAULT_DRAG_TIME_SCALE
        if path:
            with open(path, encoding="utf-8") as f:
                fit = json.load(f)
            self.samples = {k: v for k, v in fit["samples"].items() if len(v) >= MIN_SAMPLES}
            self.gestures = [np.array(g) for g in fit.get("gestures", [])]
            self.drag_time_scale = fit.get("drag_time_scale", DEFAULT_DRAG_TIME_SCALE)

    def ms(self, name):
        """One draw, in ms: a recorded value with a little jitter, or the lognormal default."""
        s = self.samples.get(name)
        if s:
            return float(self.rng.choice(s)) * float(self.rng.lognormal(0, 0.08))
        return float(DEFAULT_MEDIANS[name] * self.rng.lognormal(0, SIGMA))

    def gesture(self, total):
        """A flick moving about `total` px (positive = down). Returns [[t_ms, dx, dy]]."""
        sign, total = (1.0 if total >= 0 else -1.0), abs(total)
        if len(self.gestures) >= MIN_SAMPLES:
            # Prefer recorded flicks of similar size so the shape matches the distance.
            sizes = np.array([g[:, 2].sum() for g in self.gestures])
            w = np.exp(-np.abs(np.log(sizes / max(total, 1.0))))
            g = self.gestures[self.rng.choice(len(sizes), p=w / w.sum())].copy()
            g[:, 1:] *= total / g[:, 2].sum()
        else:
            g = synthetic_gesture(self.rng, total)
        g[:, 2] *= sign
        g[:, 1] *= self.rng.choice([-1.0, 1.0])
        return g


# ---------------------------------------------------------------- fitting from recordings


def _still_after(events, thresh=1.5):
    """Time the cursor stopped moving, relative to events[0] (events are [t, x, y])."""
    ev = np.asarray(events, float)
    moved = np.nonzero(np.linalg.norm(ev[:, 1:3] - ev[-1, 1:3], axis=1) > thresh)[0]
    return ev[moved[-1] + 1, 0] - ev[0, 0] if len(moved) else 0.0


def _moved_after(events, thresh=2.0):
    ev = np.asarray(events, float)
    moved = np.nonzero(np.linalg.norm(ev[:, 1:3] - ev[0, 1:3], axis=1) > thresh)[0]
    return ev[moved[0], 0] - ev[0, 0] if len(moved) else None


def split_gestures(wheel, gap_ms=150):
    """Wheel events [t, dx, dy, mode, x, y] -> list of [[t_rel, dx, dy]] flicks, dy made positive."""
    ev = [w for w in wheel if w[3] == 0]  # pixel mode = trackpad; line/page mode is a mouse wheel
    out, cur = [], []
    for w in ev:
        if cur and w[0] - cur[-1][0] > gap_ms:
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    gestures, pauses = [], []
    for i, g in enumerate(out):
        a = np.array([[w[0], w[1], w[2]] for w in g], float)
        if i:
            pauses.append(float(a[0, 0] - out[i - 1][-1][0]))
        if len(a) < 3 or abs(a[:, 2].sum()) < 20:
            continue
        if a[:, 2].sum() < 0:
            a[:, 1:] *= -1
        a[:, 0] -= a[0, 0]
        gestures.append(a.round(2).tolist())
    return gestures, pauses


def fit_behavior(tasks, click_trials=(), step_ms=10.0):
    s = {k: [] for k in DEFAULT_MEDIANS}
    gestures, drag_ratio = [], []

    fitts = None
    if click_trials:
        from .data import trial_points, trial_width
        from .metrics import fitts_fit

        rows = [(trial_points(t, step_ms), trial_width(t)) for t in click_trials]
        rows = [(p, w) for p, w in rows if p is not None]
        if len(rows) >= 20:
            a, b, _ = fitts_fit([(len(p) - 1) * step_ms for p, _ in rows], [np.hypot(*(p[-1] - p[0])) for p, _ in rows], [w for _, w in rows])
            fitts = (a, b)

    for t in tasks:
        kind = t["kind"]
        if kind == "drag" and "release" in t and t.get("success"):
            ev = t["events"]
            press = _moved_after(ev)
            if press is None:
                continue
            s["drag_press_ms"].append(press)
            s["drag_settle_ms"].append(t["release"][0] - ev[0][0] - _still_after(ev))
            if fitts:
                pts = trim_onset(resample(ev, step_ms))
                moving_ms = _still_after(ev) - press
                dist = float(np.hypot(*(pts[-1] - pts[0])))
                width = min(t["target_size"])  # success = the square's center lands in the box
                expected = fitts[0] + fitts[1] * math.log2(dist / width + 1)
                if dist > 50 and expected > 100 and moving_ms > 0:
                    drag_ratio.append(moving_ms / expected)
        elif kind == "scroll" and t.get("wheel"):
            g, pauses = split_gestures(t["wheel"])
            gestures += g
            s["scroll_pause_ms"] += [p for p in pauses if p < 1500]

    return {
        "samples": {k: [round(float(x), 1) for x in v if x > 0] for k, v in s.items()},
        "gestures": gestures,
        "drag_time_scale": float(np.median(drag_ratio)) if len(drag_ratio) >= MIN_SAMPLES else DEFAULT_DRAG_TIME_SCALE,
    }
