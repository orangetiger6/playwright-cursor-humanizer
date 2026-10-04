import numpy as np
import pytest

from humanmouse.behavior import DEFAULT_MEDIANS, Behavior, split_gestures, synthetic_gesture


def test_defaults_are_positive_and_near_median():
    b = Behavior(rng=np.random.default_rng(0))
    for name, median in DEFAULT_MEDIANS.items():
        draws = [b.ms(name) for _ in range(400)]
        assert min(draws) > 0
        assert 0.8 * median < np.median(draws) < 1.25 * median


@pytest.mark.parametrize("total", [120.0, 900.0, -600.0])
def test_gesture_moves_the_requested_distance(total):
    g = Behavior(rng=np.random.default_rng(1)).gesture(total)
    assert g[:, 2].sum() == pytest.approx(total)
    assert np.all(np.diff(g[:, 0]) > 0)


def test_synthetic_gesture_decays():
    g = synthetic_gesture(np.random.default_rng(2), 500)
    peak = int(np.argmax(g[:, 2]))
    assert np.all(np.diff(g[peak:, 2]) <= 1e-9)


def test_split_gestures_on_pauses():
    wheel = [[t, 0, 10, 0, 0, 0] for t in range(0, 200, 16)] + [[t, 0, -8, 0, 0, 0] for t in range(600, 800, 16)]
    wheel.append([900, 0, 3, 1, 0, 0])  # line mode (mouse wheel): ignored
    gestures, pauses = split_gestures(wheel)
    assert len(gestures) == 2 and len(pauses) == 1
    assert all(np.array(g)[:, 2].sum() > 0 for g in gestures)  # made positive
    assert pauses[0] == pytest.approx(600 - 192)

