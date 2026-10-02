"""Kinematic statistics used to compare real and generated trajectories."""

import numpy as np

from .data import canonicalize

METRICS = ["duration_ms", "path_efficiency", "max_deviation", "overshoot", "peak_speed_time", "submovements"]


def trajectory_metrics(pts, step_ms):
    q, _ = canonicalize(pts, 1.0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    speed = np.convolve(seg, np.ones(3) / 3, mode="same")
    peak = speed.max()
    interior = speed[1:-1]
    is_peak = (interior > speed[:-2]) & (interior >= speed[2:]) & (interior > 0.1 * peak)
    return {
        "duration_ms": (len(pts) - 1) * step_ms,
        "path_efficiency": float(np.hypot(*(pts[-1] - pts[0])) / max(seg.sum(), 1e-9)),
        "max_deviation": float(np.abs(q[:, 1]).max()),
        "overshoot": float(max(q[:, 0].max() - 1.0, 0.0)),
        "peak_speed_time": float(np.argmax(speed) / len(speed)),
        "submovements": int(is_peak.sum()),
    }


def fitts_fit(durations, dists, widths):
    """Least-squares MT = a + b * log2(D/W + 1). Returns (a, b, r)."""
    ids = np.log2(np.asarray(dists) / np.asarray(widths) + 1)
    b, a = np.polyfit(ids, durations, 1)
    r = float(np.corrcoef(ids, durations)[0, 1])
    return float(a), float(b), r


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))


def summarize(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in METRICS}
