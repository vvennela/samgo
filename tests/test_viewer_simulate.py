"""The viewer's Simulate data: which drawn parts turn, about which pivot, in the page's mm y-up frame."""

import json

from samgo.technic import PLATE, STUD
from samgo.viewer3d import motions, page, scene


def tv():
    return [{"type": "plate", "w": 4, "l": 4, "x": 0, "y": 0, "z": 0, "color": "light_gray"},
            {"type": "turntable_base", "w": 2, "l": 2, "x": 1, "y": 1, "z": 1, "color": "dark_gray"},
            {"type": "turntable_top", "w": 2, "l": 2, "x": 1, "y": 1, "z": 2, "color": "dark_gray"},
            {"type": "brick", "w": 6, "l": 1, "x": -1, "y": 1, "z": 3, "color": "black"}]


def test_turntable_turns_the_screen_about_a_vertical_page_axis():
    (j,) = motions(tv())
    assert j["moving"] == [2, 3] and not j["spin"]
    assert j["axis"] == [0.0, 1.0, 0.0]                       # LEGO z up is the page's y
    assert j["point"] == [2 * STUD, 1 * PLATE, 2 * STUD]       # turntable centre, (x, height, y)
    assert abs(j["degrees"]) == 360                           # nothing in the way


def test_lid_hinge_axis_runs_along_x_and_records_name_their_parts():
    parts = [{"type": "plate", "w": 4, "l": 4, "x": 0, "y": 0, "z": 0, "color": "gray"},
             {"type": "click_a", "w": 4, "l": 1, "x": 0, "y": 3, "z": 1, "pin_x": 0, "pin_y": 4, "pin_axis": "x"},
             {"type": "click_b", "w": 4, "l": 1, "x": 0, "y": 4, "z": 1, "pin_x": 0, "pin_y": 4, "pin_axis": "x"},
             {"type": "brick", "w": 4, "l": 1, "x": 0, "y": 4, "z": 2, "color": "black"}]
    (j,) = motions(parts)
    assert j["moving"] == [2, 3] and j["axis"] == [1.0, 0.0, 0.0]
    assert 0 < abs(j["degrees"]) < 360                        # the lid would hit the base going all the way round
    assert [b[-1] for b in scene(parts)["boxes"]] == [0, 1, 2, 3]


def test_page_embeds_joints_and_a_model_without_joints_has_none():
    html = page([("tv", tv(), []), ("brick", [{"type": "brick", "w": 2, "l": 2, "x": 0, "y": 0, "z": 0}], [])])
    data = json.loads(html.split("const MODELS = ", 1)[1].split(";\n", 1)[0])
    assert len(data[0]["joints"]) == 1 and data[1]["joints"] == []
