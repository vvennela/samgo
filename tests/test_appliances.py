import numpy as np

from samgo.appliances import build_laptop, build_tv


def ok(v):
    return v if isinstance(v, bool) else v[0]


def test_tv_and_laptop_pass_every_check_at_several_sizes():
    for shape in ((16, 4, 30), (24, 3, 40), (8, 2, 20)):
        r = build_tv(np.ones(shape))
        assert all(ok(v) for v in r["checks"].values()), r["checks"]
    for shape in ((12, 8, 10), (20, 12, 10), (6, 4, 6)):
        r = build_laptop(np.ones(shape))
        assert all(ok(v) for v in r["checks"].values()), r["checks"]


def test_the_lid_hinge_is_the_clip_and_bar_hinge():
    types = {p["type"] for p in build_laptop()["parts"]}
    assert {"bar_plate", "clip_plate"} <= types and "click_a" not in types
