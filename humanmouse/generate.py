"""Turn a trained model into screen-space mouse paths.

Sampling runs in numpy: the bundled model ships as plain arrays (assets/model.npz), so PyTorch is
only needed for training. A training checkpoint (.pt) loads too when torch is installed, and
`export_npz` turns one into the numpy format.
"""
import json
import math
from pathlib import Path

import numpy as np

from .data import MAX_STEPS, MIN_DIST_PX, condition_vector, uncanonicalize

DEFAULT_MODEL = Path(__file__).parent / "assets" / "model.npz"  # SapiMouse, fine-tuned on one person


def load_checkpoint(path):
    """(weights, config, stats) from a numpy export (.npz) or a training checkpoint (.pt, needs torch)."""
    path = str(path)
    if path.endswith(".pt"):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return {k: v.float().numpy() for k, v in ckpt["model"].items()}, ckpt["config"], ckpt["stats"]
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        return {k: f[k] for k in f.files if k != "meta"}, meta["config"], meta["stats"]


def export_npz(checkpoint, out):
    """Save a training checkpoint as the numpy format that `PathGenerator` loads without torch."""
    weights, config, stats = load_checkpoint(checkpoint)
    np.savez_compressed(out, meta=json.dumps({"config": config, "stats": stats}), **weights)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _erf(x):  # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    s, x = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return s * (1.0 - poly * np.exp(-x * x))


class MDN:
    """MouseMDN's forward pass (input layer, GRU, mixture head) one step at a time, in numpy."""

    def __init__(self, weights, config):
        self.w = {k: v.astype(np.float64) for k, v in weights.items()}
        self.layers, self.m, self.hidden = config["layers"], config["mixtures"], config["hidden"]

    def step(self, x, h):
        """One input vector -> (head output, new hidden state); `h` is None at the start."""
        w = self.w
        h = [np.zeros(self.hidden)] * self.layers if h is None else h
        z = w["inp.0.weight"] @ x + w["inp.0.bias"]
        z = 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))  # exact GELU, as nn.GELU()
        new_h = []
        for i in range(self.layers):  # torch.nn.GRU, gates in r, z, n order
            ir, iz, in_ = np.split(w[f"rnn.weight_ih_l{i}"] @ z + w[f"rnn.bias_ih_l{i}"], 3)
            hr, hz, hn = np.split(w[f"rnn.weight_hh_l{i}"] @ h[i] + w[f"rnn.bias_hh_l{i}"], 3)
            r, u = _sigmoid(ir + hr), _sigmoid(iz + hz)
            z = (1.0 - u) * np.tanh(in_ + r * hn) + u * h[i]
            new_h.append(z)
        return w["head.weight"] @ z + w["head.bias"], new_h

    def sample(self, out, temperature, rng):
        """One (delta, end) draw from the head output, like MouseMDN.sample_step."""
        m = self.m
        logit_pi, mu, log_sigma, rho, end = np.split(out, [m, 3 * m, 5 * m, 6 * m])
        p = np.exp((logit_pi - logit_pi.max()) / temperature)
        k = rng.choice(m, p=p / p.sum())
        mu, rho = mu.reshape(m, 2)[k], np.tanh(rho[k]) * 0.95
        sigma = np.exp(np.clip(log_sigma.reshape(m, 2)[k], -6.0, 3.0)) * math.sqrt(temperature)
        e = rng.standard_normal(2)
        delta = np.array([mu[0] + sigma[0] * e[0], mu[1] + sigma[1] * (rho * e[0] + math.sqrt(1 - rho**2) * e[1])])
        return delta, rng.random() < _sigmoid(end[0])


def min_jerk(start, end, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    return start + (end - start) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def snap_endpoint(q):
    """Bend a canonical path so it ends exactly at (1, 0), weighting the fix toward the end."""
    err = np.array([1.0, 0.0]) - q[-1]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    w = arc / arc[-1] if arc[-1] > 0 else np.linspace(0.0, 1.0, len(q))
    w = 3 * w**2 - 2 * w**3
    return q + w[:, None] * err


class PathGenerator:
    def __init__(self, checkpoint=None, seed=None):
        weights, config, stats = load_checkpoint(checkpoint or DEFAULT_MODEL)
        self.model = MDN(weights, config)
        self.scale = np.asarray(stats["delta_scale"], dtype=np.float64)
        self.step_ms = stats["step_ms"]
        self.rng = np.random.default_rng(seed)

    def sample_canonical(self, cond, temperature=0.8, max_steps=MAX_STEPS):
        cond = np.asarray(cond, dtype=np.float64)
        goal = np.array([1.0, 0.0])
        prev, pos, h = np.zeros(2), np.zeros(2), None
        out = [pos]
        for _ in range(max_steps):
            params, h = self.model.step(np.concatenate([prev, pos - goal, cond]), h)
            prev, end = self.model.sample(params, temperature, self.rng)
            pos = pos + prev * self.scale
            out.append(pos)
            if end:
                break
        return np.array(out, dtype=np.float64)

    def path(self, start, end, target_width=20.0, temperature=0.8, tries=4):
        """Returns (points [N, 2], times_ms [N]) from start to end, inclusive."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        dist = float(np.hypot(*(end - start)))
        if dist < MIN_DIST_PX:
            # Too short for the model's training range: small minimum-jerk hop.
            n = max(3, int((60 + 25 * math.log2(dist / max(target_width, 1.0) + 1)) / self.step_ms))
            pts = min_jerk(start, end, n)
        else:
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            cond = condition_vector(dist, target_width, angle)
            best, best_err = None, math.inf
            for _ in range(tries):
                q = self.sample_canonical(cond, temperature)
                err = float(np.hypot(q[-1, 0] - 1.0, q[-1, 1]))
                if err < best_err:
                    best, best_err = q, err
                if err < 0.05:
                    break
            pts = uncanonicalize(snap_endpoint(best), start, end)
        return pts, np.arange(len(pts)) * self.step_ms

    def point_in_box(self, x, y, w, h):
        """Humans aim near the center but not at it; clipped to stay inside the element."""
        px = self.rng.normal(x + w / 2, w / 6)
        py = self.rng.normal(y + h / 2, h / 6)
        return float(np.clip(px, x + 2, x + w - 2)), float(np.clip(py, y + 2, y + h - 2))
