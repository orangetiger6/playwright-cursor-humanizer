import numpy as np

from humanmouse.data import canonicalize, resample, trim_onset, uncanonicalize


def test_resample_grid_and_hold():
    ev = [[0, 0, 0], [10, 10, 0], [200, 20, 0]]  # a pause before the last event
    pts = resample(ev, step_ms=10)
    assert len(pts) == 21
    assert pts[-1].tolist() == [20, 0]
    assert pts[10].tolist() == [10, 0]  # held during the pause, not drifting


def test_trim_onset():
    pts = np.array([[0, 0]] * 5 + [[1, 0], [5, 0], [10, 0]], float)
    assert len(trim_onset(pts)) == 3


def test_canonical_round_trip():
    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.normal(3, 2, (50, 2)), axis=0) + [100, 200]
    q, cond = canonicalize(pts, 20)
    assert np.allclose(q[0], [0, 0]) and np.allclose(q[-1], [1, 0])
    assert np.allclose(uncanonicalize(q, pts[0], pts[-1]), pts)
