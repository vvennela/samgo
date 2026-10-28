import numpy as np

from samgo.car import _segments, build_car
from samgo.parts import COLOR_INDEX


def test_chassis_split_keeps_axles_inside_technic_bricks():
    plan = _segments(0, 20, [3, 17])
    assert sum(n for _, n, _ in plan) == 20
    for e in (3, 17):
        assert any(t and s < e < s + n for s, n, t in plan)


def test_car_target_builds_into_a_car_that_rolls():
    t = np.zeros((12, 6, 9), dtype=np.int32)
    t[:, :, :5] = COLOR_INDEX["red"]
    t[3:9, :, 5:8] = COLOR_INDEX["blue"]
    car = build_car(t)
    assert car.structure.passed and car.function.passed
    assert int((car.target[:, :, 3:] > 0).sum()) == int((t > 0).sum())  # lifted: nothing of the shape lost


def test_car_longer_in_y_is_turned_and_too_wide_is_refused():
    t = np.zeros((6, 12, 9), dtype=np.int32)
    t[:, :, :6] = COLOR_INDEX["red"]
    assert build_car(t).function.passed
    assert build_car(np.full((20, 14, 6), COLOR_INDEX["red"], dtype=np.int32)) is None
