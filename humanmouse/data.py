"""Loading, cleaning and canonicalizing recorded mouse trajectories.

A trajectory is resampled onto a fixed time grid, then expressed in a canonical
frame: start at (0, 0), end at (1, 0). Distance, target width and direction are
kept as a conditioning vector so the model can learn how they change the motion.
"""
import glob
import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset

STEP_MS = 10.0
HOLD_GAP_MS = 50.0
MIN_DIST_PX = 20.0
MAX_STEPS = 400  # 4 s at 10 ms; longer trials are usually the user getting distracted
COND_DIM = 4


def load_trials(pattern, sources=None):
    trials = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            session = json.load(f)
        if sources and session.get("source", "human") not in sources:
            continue
        # Pointing movements only; drag / type / scroll task records are read by humanmouse.behavior.
        trials.extend(t for t in session["trials"] if "click" in t and t.get("kind", "click") == "click")
    return trials


def resample(events, step_ms=STEP_MS, hold_gap_ms=HOLD_GAP_MS):
    """Linear-interpolate irregular [t, x, y] events onto a fixed grid. Keeps the final event."""
    ev = np.asarray(events, dtype=np.float64)
    t = ev[:, 0] - ev[0, 0]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, xy = t[keep], ev[keep, 1:3]
    # Event-driven logs are silent while the cursor is still; plain interpolation would smear a
    # pause into slow drift. Hold the last position until just before the next event.
    gaps = np.nonzero(np.diff(t) > hold_gap_ms)[0]
    if len(gaps):
        t = np.insert(t, gaps + 1, t[gaps + 1] - step_ms)
        xy = np.insert(xy, gaps + 1, xy[gaps], axis=0)
    grid = np.append(np.arange(0.0, t[-1], step_ms), t[-1])
    return np.stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])], axis=1)


def trim_onset(pts, thresh_px=2.0):
    """Drop the reaction-time period before the cursor starts moving."""
    moved = np.nonzero(np.linalg.norm(pts - pts[0], axis=1) > thresh_px)[0]
    return pts if len(moved) == 0 else pts[max(moved[0] - 1, 0):]


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def condition_vector(dist, width, angle):
    return np.array(
        [math.log(dist / 100.0), math.log(max(width, 1.0) / dist), math.sin(angle), math.cos(angle)],
        dtype=np.float32,
    )


def canonicalize(pts, width):
    start, end = pts[0], pts[-1]
    v = end - start
    dist = float(np.hypot(*v))
    angle = math.atan2(v[1], v[0])
    q = (pts - start) @ rotation(-angle).T / dist
    return q, condition_vector(dist, width, angle)


def uncanonicalize(q, start, end):
    start, end = np.asarray(start, float), np.asarray(end, float)
    v = end - start
    return start + float(np.hypot(*v)) * q @ rotation(math.atan2(v[1], v[0])).T


def trial_points(trial, step_ms=STEP_MS):
    """Resampled, onset-trimmed screen-space points for a trial, or None if unusable."""
    if len(trial["events"]) < 3:
        return None
    pts = trim_onset(resample(trial["events"], step_ms))
    if len(pts) < 3 or len(pts) > MAX_STEPS + 1:
        return None
    if np.hypot(*(pts[-1] - pts[0])) < MIN_DIST_PX:
        return None
    return pts


def trial_width(trial):
    return float(min(trial["target_size"]))


def prepare(trial, step_ms=STEP_MS):
    pts = trial_points(trial, step_ms)
    if pts is None:
        return None
    q, cond = canonicalize(pts, trial_width(trial))
    if np.abs(q).max() > 3.0:  # tracking glitches / wild detours
        return None
    return q.astype(np.float32), cond


class TrajectoryDataset(Dataset):
    """Teacher-forcing samples.

    Input at step t:  [previous delta (scaled), position relative to target, condition]
    Target at step t: delta_t (scaled) and an end-of-movement flag.
    """

    def __init__(self, trials, stats=None, step_ms=STEP_MS, jitter_px=0.0):
        self.items = [it for it in (prepare(t, step_ms) for t in trials) if it is not None]
        # Uniform +-jitter_px noise on interior points ("dequantization"). Integer pixel logs and
        # held pauses give exact-zero deltas, which let the mixture collapse onto spikes.
        self.jitter_px = jitter_px
        self.rng = np.random.default_rng()
        if not self.items:
            raise ValueError("no usable trials")
        if stats is None:
            deltas = np.concatenate([np.diff(q, axis=0) for q, _ in self.items])
            stats = {"delta_scale": deltas.std(axis=0).tolist(), "step_ms": step_ms}
        self.stats = stats
        self.scale = np.asarray(stats["delta_scale"], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        q, cond = self.items[i]
        if self.jitter_px > 0:
            dist = 100.0 * math.exp(float(cond[0]))
            q = q.copy()
            q[1:-1] += self.rng.uniform(-self.jitter_px, self.jitter_px, q[1:-1].shape) / dist
        d = np.diff(q, axis=0) / self.scale
        prev = np.vstack([np.zeros((1, 2), np.float32), d[:-1]])
        rel = q[:-1] - np.array([1.0, 0.0], np.float32)
        x = np.concatenate([prev, rel, np.broadcast_to(cond, (len(d), COND_DIM))], axis=1)
        end = np.zeros(len(d), np.float32)
        end[-1] = 1.0
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(d.astype(np.float32)), torch.from_numpy(end)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batches of similar-length trajectories, in random order, to avoid padding waste."""

    def __init__(self, dataset, batch_size, shuffle=True, pool=50, seed=0):
        self.lengths = [len(q) for q, _ in dataset.items]
        self.batch_size, self.shuffle, self.pool = batch_size, shuffle, pool
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        chunk = self.batch_size * self.pool
        batches = []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i : i + chunk], key=lambda j: self.lengths[j])
            batches += [part[k : k + self.batch_size] for k in range(0, len(part), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return -(-len(self.lengths) // self.batch_size)


def collate(batch):
    n = max(len(x) for x, _, _ in batch)
    xs = torch.zeros(len(batch), n, batch[0][0].shape[1])
    ds = torch.zeros(len(batch), n, 2)
    ends = torch.zeros(len(batch), n)
    mask = torch.zeros(len(batch), n)
    for i, (x, d, e) in enumerate(batch):
        xs[i, : len(x)], ds[i, : len(d)], ends[i, : len(e)], mask[i, : len(x)] = x, d, e, 1.0
    return xs, ds, ends, mask
