"""Clip-and-bar hinge (60478 bar plate + 63868 clip plate): a horizontal hinge on the stud grid."""

import numpy as np
import pytest

from samgo.kinematics import analyse, connected, locked, requirement_met
from samgo.mechanism_bases import PIVOT_AS
from samgo.stability import stability


def flap():
    """A wing (2 plates thick) with a bar plate at its rear edge facing +x; a flap carrying the clip plate one stud
    behind, facing -x. The wing stands on a 2x4 brick, so the whole thing rests on the ground."""
    wing = [{"type": "brick", "w": 4, "l": 2, "x": 0, "y": 0, "z": 0, "color": "white"},
            {"type": "plate", "w": 4, "l": 2, "x": 0, "y": 0, "z": 3, "color": "white"},
            {"type": "bar_plate", "w": 2, "l": 1, "x": 2, "y": 0, "z": 4, "facing": "+x", "color": "dark_gray"},
            {"type": "bar_plate", "w": 2, "l": 1, "x": 2, "y": 1, "z": 4, "facing": "+x", "color": "dark_gray"},
            {"type": "plate", "w": 2, "l": 2, "x": 0, "y": 0, "z": 4, "color": "white"}]
    surface = [{"type": "clip_plate", "w": 2, "l": 1, "x": 5, "y": 0, "z": 4, "facing": "-x", "color": "light_gray"},
               {"type": "clip_plate", "w": 2, "l": 1, "x": 5, "y": 1, "z": 4, "facing": "-x", "color": "light_gray"},
               {"type": "plate", "w": 2, "l": 2, "x": 5, "y": 0, "z": 5, "color": "light_gray"}]
    return wing + surface


def test_the_flap_swings_about_a_horizontal_axis_and_holds():
    parts = flap()
    bodies, joints = analyse(parts)
    assert any(abs(j[4][1]) > 0.99 for j in joints)                      # a joint about the y axis
    assert connected(parts)
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "horizontal", "degrees": 30, "count": 1})
    assert ok, why
    assert stability(locked(parts, PIVOT_AS)).passed


def test_without_the_gap_there_is_no_hinge():
    parts = flap()
    for p in parts[5:]:
        p["x"] -= 1                                                      # clip butts against the bar plate
    _, joints = analyse(parts)
    assert not any(abs(j[4][1]) > 0.99 for j in joints)


pytest.importorskip("samgo.ldraw")
from samgo.ldraw import LIB, place, points  # noqa: E402


@pytest.mark.skipif(not (LIB / "parts" / "60478.dat").exists(), reason="LDraw library cache not downloaded")
def test_exported_bar_and_clip_meet_in_the_real_geometry():
    parts = flap()
    lines = [place(parts[2], "60478")[0], place(parts[5], "63868")[0]]
    centres = []
    for line, local in zip(lines, (np.array([30.0, 2, 0]), np.array([30.0, 2, 0]))):  # bar / clip centre, LDU
        t = line.split()
        pos, rot = np.array(list(map(float, t[2:5]))), np.array(list(map(float, t[5:14]))).reshape(3, 3)
        centres.append(rot @ local + pos)
    assert np.allclose(centres[0], centres[1])
