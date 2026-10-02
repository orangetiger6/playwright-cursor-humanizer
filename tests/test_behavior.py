import numpy as np
import pytest

from humanmouse.behavior import DEFAULT_MEDIANS, Behavior, fit_behavior, gap_kind, split_gestures, synthetic_gesture


@pytest.mark.parametrize("prev, ch, kind", [
    ("a", " ", "space"), (" ", "b", "word"), ("a", "B", "shift"), ("a", "!", "shift"),
    ("a", ",", "punct"), ("a", "b", "char"), ("1", "2", "char"),
])
def test_gap_kind(prev, ch, kind):
    assert gap_kind(prev, ch) == kind


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


def test_fit_behavior_from_typing():
    keys, t = [], 1000.0
    for ch in "hello world":
        keys += [[t, "down", ch, f"Key{ch}"], [t + 90, "up", ch, f"Key{ch}"]]
        t += 150
    task = {"kind": "type", "focus": [500.0, 0, 0], "keys": keys, "resume_ms": 600}
    s = fit_behavior([task] * 3)["samples"]
    assert s["homing_ms"] == [500.0] * 3
    assert set(s["hold_ms"]) == {90.0} and set(s["gap_char_ms"]) == {150.0}
    assert len(s["gap_space_ms"]) == 3 and len(s["gap_word_ms"]) == 3
