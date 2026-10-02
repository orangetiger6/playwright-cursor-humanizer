"""Turn a trained checkpoint into screen-space mouse paths."""
import math

import numpy as np
import torch

from .data import MAX_STEPS, MIN_DIST_PX, condition_vector, uncanonicalize
from .model import MouseMDN


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
    def __init__(self, checkpoint, seed=None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = MouseMDN(**ckpt["config"]).eval()
        self.model.load_state_dict(ckpt["model"])
        self.scale = torch.tensor(ckpt["stats"]["delta_scale"], dtype=torch.float32)
        self.step_ms = ckpt["stats"]["step_ms"]
        self.torch_gen = torch.Generator().manual_seed(seed if seed is not None else torch.seed() % 2**63)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def sample_canonical(self, cond, temperature=0.8, max_steps=MAX_STEPS):
        cond = torch.as_tensor(cond, dtype=torch.float32)
        goal = torch.tensor([1.0, 0.0])
        prev, pos, h = torch.zeros(2), torch.zeros(2), None
        out = [pos.numpy().copy()]
        for _ in range(max_steps):
            x = torch.cat([prev, pos - goal, cond]).view(1, 1, -1)
            params, h = self.model(x, h)
            params = {k: v[0, 0] for k, v in params.items()}
            prev, end = self.model.sample_step(params, temperature, self.torch_gen)
            pos = pos + prev * self.scale
            out.append(pos.numpy().copy())
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
