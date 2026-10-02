import numpy as np
import pytest


@pytest.mark.parametrize("start, end, width", [((100, 100), (700, 400), 30), ((500, 500), (480, 150), 10)])
def test_path_ends_on_target(generator, start, end, width):
    pts, times = generator.path(start, end, width)
    assert np.allclose(pts[0], start) and np.allclose(pts[-1], end)
    assert len(pts) == len(times) and np.all(np.diff(times) > 0)
    assert 80 < times[-1] < 4000  # a plausible human movement time


def test_point_in_box_stays_inside(generator):
    for _ in range(200):
        x, y = generator.point_in_box(10, 20, 40, 30)
        assert 10 <= x <= 50 and 20 <= y <= 50
