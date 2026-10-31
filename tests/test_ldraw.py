"""LDraw export: part numbers for every type the builders emit, and placement against the real library geometry
(skipped when the downloaded library cache is absent, so the suite stays offline)."""

import numpy as np
import pytest

from samgo.ldraw import LIB, color_code, export, part_ids, place, points

TYPES = [
    {"type": "brick", "w": 2, "l": 4, "x": 0, "y": 0, "z": 0, "color": "red"},
    {"type": "plate", "w": 1, "l": 6, "x": 3, "y": 1, "z": 3},
    {"type": "technic_brick", "w": 4, "l": 1, "x": 0, "y": 2, "z": 0, "hole": "round"},
    {"type": "axle", "axis": "y", "x": 2, "y": 0, "z": 0, "length": 8},
    {"type": "axle", "axis": "y", "x": 2, "y": 0, "z": 0, "length": 4, "stop": True},
    {"type": "wheel", "axis": "y", "x": 2, "y": 0, "z": 0, "size": "30.4x14"},
    {"type": "bush", "axis": "y", "x": 2, "y": 0, "z": 0},
    {"type": "half_bush", "axis": "y", "x": 2, "y": 0, "z": 0},
    {"type": "pin", "axis": "y", "x": 2, "y": 0, "z": 0, "length": 2, "friction": True},
    {"type": "liftarm", "axis": "x", "x": 0, "y": 0, "z": 0, "length": 7},
    {"type": "turntable_base", "w": 2, "l": 2, "x": 0, "y": 0, "z": 0},
    {"type": "turntable_top", "w": 2, "l": 2, "x": 0, "y": 0, "z": 1},
    {"type": "hinge_base", "w": 2, "l": 1, "x": 0, "y": 0, "z": 0, "pin_x": 2, "pin_y": 0},
    {"type": "click_a", "w": 2, "l": 1, "x": 0, "y": 0, "z": 0, "pin_x": 2, "pin_y": 1, "pin_axis": "x"},
]


def test_every_builder_part_type_has_an_ldraw_part_and_colour():
    for p in TYPES:
        assert part_ids(p), p
        assert isinstance(color_code(p), int)
    assert part_ids(TYPES[0]) == ["3001"] and part_ids(TYPES[5]) == ["6578", "6014"]


have_lib = pytest.mark.skipif(not (LIB / "parts" / "3001.dat").exists(), reason="LDraw library cache not downloaded")


@have_lib
def test_a_brick_lands_exactly_where_the_checker_puts_it():
    line, err = place(TYPES[0], "3001")
    tok = line.split()
    t, R = np.array(list(map(float, tok[2:5]))), np.array(list(map(float, tok[5:14]))).reshape(3, 3)
    pts = points("3001.dat") @ R.T + t
    # ours: x 0..2, y 0..4 studs, z 0..3 plates -> LDraw X 0..40, Z 0..80, Y 0..-24 (up is -Y)
    assert np.allclose(pts.min(0), [0, -24, 0]) and np.allclose(pts.max(0), [40, 0, 80])
    assert err < 1e-9


@have_lib
def test_an_axle_runs_along_our_axis_through_the_hole_line():
    line, err = place(TYPES[3], "3707")
    tok = line.split()
    t, R = np.array(list(map(float, tok[2:5]))), np.array(list(map(float, tok[5:14]))).reshape(3, 3)
    pts = points("3707.dat") @ R.T + t
    span = pts.max(0) - pts.min(0)
    assert np.argmax(span) == 2                       # along LDraw Z = our y
    assert abs(t[0] - 40) < 1e-6                     # on stud edge x = 2
    assert abs(t[1] - -14) < 1e-6                    # hole height 5.6 mm = 14 LDU up
    assert err < 0.5


@have_lib
def test_export_reports_parts_whose_real_size_differs_from_ours():
    _, report = export([TYPES[0], TYPES[13]])
    by_type = {r["type"]: r["mismatch_mm"] for r in report}
    assert by_type["brick"] < 1e-9
    assert by_type["click_a"] > 1                    # the real locking hinge has a finger past its 1x2 plate
