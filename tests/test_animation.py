import glob
import json
import random

import numpy as np

from samgo.animation import order, paths, slide_clear, through_solid


def test_slide_clear_matches_a_brute_force_sweep():
    """formal/AnimationPath.lean proves slide_clear; here the Python condition is checked against moving the box."""
    rng = random.Random(0)

    def rbox():
        lo = [rng.randint(0, 20) for _ in range(3)]
        return lo, [v + rng.randint(1, 6) for v in lo]

    def collide(a, b):
        return all(a[0][k] < b[1][k] and b[0][k] < a[1][k] for k in range(3))

    for _ in range(3000):
        m, p = rbox(), rbox()
        axis, sign = rng.randint(0, 2), rng.choice((1, -1))
        if collide(m, p) or not slide_clear(m, p, axis, sign):
            continue
        for t in range(0, 40):
            lo, hi = list(m[0]), list(m[1])
            lo[axis] += sign * t
            hi[axis] += sign * t
            assert not collide((lo, hi), p)


def test_every_part_of_a_car_has_a_clear_path_and_the_chassis_comes_first():
    paths_ = sorted(glob.glob("runs/demo_buildable/camry/artifacts/*/build.json"))
    if not paths_:
        return
    parts = json.load(open(paths_[-1]))
    steps = order(parts)
    _, blocked = paths(parts, steps)
    assert not blocked
    first = [parts[i]["type"] for i in sorted(range(len(parts)), key=lambda i: steps[i])]
    k = first.index("plate")
    assert set(first[:k]) <= {"technic_brick", "axle", "wheel", "bush", "half_bush"} and "axle" in first[:k]


def test_an_axle_through_a_solid_brick_is_found():
    from samgo.builder import build
    from samgo.parts import COLOR_INDEX

    g = np.zeros((6, 6, 6), int)
    g[:] = COLOR_INDEX["white"]
    body = build(g, attempts=1, pull=lambda p: 0.0).pieces
    axle = {"type": "axle", "axis": "y", "x": 3, "y": 0, "z": 1, "length": 4, "color": "black"}
    assert through_solid(body + [axle])
