import asyncio

import numpy as np

from samgo import spark_mechanisms as sm
from samgo import spark_review as sr
from samgo.builder import build
from samgo.edits import apply_edits
from samgo.kinematics import rotors
from samgo.parts import COLOR_INDEX
from samgo.rotor_data import ROTOR, answer


def block():
    g = np.zeros((8, 8, 15), int)
    g[:] = COLOR_INDEX["white"]
    return build(g, attempts=1, pull=lambda p: 0.0).pieces


def test_parsers_keep_only_the_fixed_vocabulary():
    regions = {1: {"part": "body"}, 2: {"part": "wing"}}
    m = sr.parse_mech('{"add": [{"region": 2, "kind": "spinning", "name": "fan", "where": "front", "axis": "horizontal"},'
                      ' {"region": 9, "kind": "spinning", "name": "x"}, {"region": 1, "kind": "teleport", "name": "y"}],'
                      ' "remove": [{"id": 3, "why": "not there"}, {"id": 99}]}', regions, {3, 4})
    assert m == {"add": [{"region": 2, "kind": "spinning", "name": "fan", "where": "front", "axis": "horizontal"}],
                 "remove": [{"id": 3, "why": "not there"}]}
    look = sr.parse_look('{"paint": [{"name": "body", "color": "red", "x": [0, 1], "y": [0, 0], "z": [2, 1]},'
                         ' {"color": "chartreuse", "x": [0, 1], "y": [0, 1], "z": [0, 1]}],'
                         ' "glass": [{"name": "w", "x": [1, 1], "y": [0, 2], "z": [3, 3]}]}', 2)
    assert look["paint"] == [{"name": "body", "color": "red", "x": [0, 3], "y": [0, 1], "z": [2, 5]}]
    assert look["glass"] == [{"name": "w", "x": [2, 3], "y": [0, 5], "z": [6, 7]}]


def test_paint_recolors_body_parts_in_the_box_only():
    body = block()
    out, n = sr.paint(body, [{"color": "red", "x": [0, 7], "y": [0, 7], "z": [0, 0]}])
    assert n > 0 and all(p["color"] == "red" for p in out if p["z"] == 0)
    assert all(p["color"] == "white" for p in out if p["z"] > 2)
    assert [{k: v for k, v in p.items() if k != "color"} for p in out] == \
           [{k: v for k, v in p.items() if k != "color"} for p in body]


def test_review_removes_a_wrong_rotor_and_adds_a_missing_one(monkeypatch, tmp_path):
    body = block()
    one, two = answer("+y", 8, 3, 6), answer("-y", 0, 3, 6)
    wrong = {"id": 101, "region": 1, "name": "rotor", "passed": True}
    passing = [(wrong, {"edits": one}, ROTOR)]
    parts = apply_edits(body, one).parts
    report = {"regions": {1: {"part": "block"}}, "surfaces": [], "rotors": [wrong], "unsupported": []}
    answers = iter([{"add": [{"region": 1, "kind": "spinning", "name": "fan", "where": "back", "axis": "horizontal"}],
                     "remove": [{"id": 101, "why": "wrong side"}]},
                    {"add": [], "remove": []}])
    monkeypatch.setattr(sr, "ask_mechanisms", lambda *a, **k: next(answers))
    built = []

    def fake_build(photo, b, obj, sampler, work, rounds, max_surfaces, labels, loop, scale, named, merge):
        assert b is body and merge is False and named["regions"][1]["spinning"][0]["name"] == "fan"
        e = {"id": 202, "region": 1, "name": "fan", "passed": True}
        built.append(e)
        return [(e, {"edits": two}, ROTOR)], {"rotors": [e]}

    monkeypatch.setattr(sm, "_build", fake_build)
    (tmp_path / "regions.png").write_bytes(b"x")
    loop = asyncio.new_event_loop()
    try:
        out = sm._review(tmp_path / "p.jpg", body, parts, "thing", None, tmp_path, 3, 8, np.zeros((8, 8, 15), int),
                         loop, 1.0, report, passing)
    finally:
        loop.close()
    assert wrong["removed_by_review"] == "wrong side" and wrong["passed"] is False
    assert built and rotors(out)[0] == 1
    assert any(p["type"] == "liftarm" and p["y"] < 0 for p in out)  # the new one is on the -y face
    assert [r["round"] for r in report["review"]] == [1, 2]


def test_render_draws_three_views(tmp_path):
    sr.render(block(), tmp_path / "v.png")
    assert (tmp_path / "v.png").stat().st_size > 1000


def test_a_part_spark_calls_wrong_stays_when_no_replacement_builds(monkeypatch, tmp_path):
    body = block()
    one = answer("+y", 8, 3, 6)
    wrong = {"id": 301, "region": 1, "name": "rotor", "passed": True}
    passing = [(wrong, {"edits": one}, ROTOR)]
    parts = apply_edits(body, one).parts
    report = {"regions": {1: {"part": "block"}}, "surfaces": [], "rotors": [wrong], "unsupported": []}
    answers = iter([{"add": [{"region": 1, "kind": "spinning", "name": "fan", "where": "x", "axis": "horizontal"}],
                     "remove": [{"id": 301, "why": "misplaced"}]},
                    {"add": [{"region": 1, "kind": "spinning", "name": "fan", "where": "x", "axis": "horizontal"}],
                     "remove": []}])
    monkeypatch.setattr(sr, "ask_mechanisms", lambda *a, **k: next(answers))
    calls = []

    def fake_build(*a, **k):  # the replacement cannot be built
        calls.append(1)
        return [], {"rotors": [{"id": 302, "region": 1, "name": "fan", "passed": False}]}

    monkeypatch.setattr(sm, "_build", fake_build)
    (tmp_path / "regions.png").write_bytes(b"x")
    loop = asyncio.new_event_loop()
    try:
        out = sm._review(tmp_path / "p.jpg", body, parts, "thing", None, tmp_path, 3, 8, np.zeros((8, 8, 15), int),
                         loop, 1.0, report, passing)
    finally:
        loop.close()
    assert rotors(out)[0] == 1 and "kept" in wrong["review_note"] and not wrong.get("removed_by_review")
    assert len(calls) == 1  # the failed request is not tried a second time


def test_base_color_then_trim_and_windows_not_on_the_roof():
    from samgo.spark_glass import apply as glaze_boxes

    body = block()
    out, _ = sr.paint(body, [{"color": "blue", "x": [-99, 99], "y": [-99, 99], "z": [-99, 99]}])
    out, _ = sr.paint(out, [{"color": "red", "x": [0, 7], "y": [0, 7], "z": [0, 0]}])
    assert {p["color"] for p in out if p["z"] == 0} == {"red"} and {p["color"] for p in out if p["z"] > 2} == {"blue"}
    top = max(p["z"] + (3 if p["type"] == "brick" else 1) for p in body) - 1
    box = {"x": [0, 7], "y": [0, 7], "z": [top - 1, top]}
    glazed, _ = glaze_boxes(body, {"glass": [box | {"name": "side windows"}]}, cut=True)
    roof = [p for p in glazed if p["color"] == "trans_clear" and p["x"] > 0 and p["x"] + p["w"] < 8
            and p["y"] > 0 and p["y"] + p["l"] < 8]
    assert not roof  # interior of the top face stays body
    _, side = glaze_boxes(body, {"glass": [box | {"name": "side windows"}]}, cut=True)
    _, canopy = glaze_boxes(body, {"glass": [box | {"name": "canopy"}]}, cut=True)
    assert canopy["cells"] > side["cells"] > 0  # a canopy may take the top face; side windows only the sides
