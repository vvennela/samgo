import numpy as np

from samgo.wheels import FALLBACK, wheel_fractions


def disc(shape, cx, cy, rx, ry):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def car_mask():
    m = np.zeros((200, 400), dtype=bool)
    m[60:170, 50:350] = True  # object spans x 50..350
    return m


def test_side_view_gives_axle_ratios():
    obj = car_mask()
    out = wheel_fractions(obj, [disc(obj.shape, 110, 160, 25, 25), disc(obj.shape, 290, 160, 25, 25)])
    assert out["source"] == "masks"
    assert abs(out["fractions"][0] - 0.2) < 0.01 and abs(out["fractions"][1] - 0.8) < 0.01


def test_hidden_far_wheel_merges_into_its_axle():
    obj = car_mask()
    wheels = [disc(obj.shape, 110, 160, 25, 25), disc(obj.shape, 120, 150, 15, 20), disc(obj.shape, 290, 160, 25, 25)]
    assert wheel_fractions(obj, wheels)["source"] == "masks"


def test_squashed_wheels_fall_back():
    obj = car_mask()
    out = wheel_fractions(obj, [disc(obj.shape, 110, 160, 10, 25), disc(obj.shape, 290, 160, 10, 25)])
    assert out["fractions"] == FALLBACK and "side view" in out["reason"]


def test_one_axle_falls_back():
    obj = car_mask()
    assert wheel_fractions(obj, [disc(obj.shape, 110, 160, 25, 25)])["source"] == "fallback"


def test_three_quarter_view_with_foreshortened_wheelbase_falls_back():
    obj = car_mask()  # Beetle photo: wheels at 9% and 57%, roundness 0.78 / 0.89
    out = wheel_fractions(obj, [disc(obj.shape, 77, 160, 20, 25), disc(obj.shape, 222, 160, 24, 27)])
    assert out["fractions"] == FALLBACK and out["source"] == "fallback"


def test_slivers_and_small_masks_are_not_wheels():
    obj = car_mask()  # truck photo: two wheels plus a thin mask (roundness 1.9) between them
    wheels = [disc(obj.shape, 134, 160, 24, 25), disc(obj.shape, 248, 160, 6, 3), disc(obj.shape, 281, 160, 21, 23)]
    out = wheel_fractions(obj, wheels)
    assert out["source"] == "masks" and len(out["axles"]) == 2
