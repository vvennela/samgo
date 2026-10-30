"""The rolling office chair: real LEGO parts, every check passes, and the motion checks can fail."""

import numpy as np

from samgo.chair import build_chair, checks
from samgo.kinematics import requirement_met


def ok(v):
    return v if isinstance(v, bool) else v[0]


def test_default_chair_passes_every_check():
    r = build_chair()
    assert {q["name"] for q in r["requirements"]} == {"casters_spin", "casters_swivel", "seat_swivels"}
    failed = {k: v for k, v in r["checks"].items() if not ok(v)}
    assert not failed, failed
    for k in ("structure", "connected", "stands", "casters_spin", "casters_swivel", "seat_swivels",
              "seat_is_moving_side"):
        assert k in r["checks"]
    assert r["unsupported"]


def test_chair_has_four_casters_and_a_seat_turntable():
    parts = build_chair()["parts"]
    n = lambda t: sum(p["type"] == t for p in parts)  # noqa: E731
    assert n("wheel") == 4 and n("turntable_base") == 5 and n("turntable_top") == 5 and n("axle") == 4
    # all five turntables give a clear 360 degree swivel (4 casters + the seat)
    assert requirement_met(parts, {"kind": "swing", "axis": "vertical", "degrees": 360, "count": 5})[0]


def test_chair_from_a_target_box_passes():
    r = build_chair(np.ones((16, 16, 50), dtype=int))
    assert all(ok(v) for v in r["checks"].values()), r["checks"]


def test_a_post_in_a_casters_path_stops_its_swivel():
    r = build_chair()
    parts = list(r["parts"])
    fork = next(p for p in parts if p["type"] == "plate" and p["w"] == 2 and p["l"] == 4)
    # a brick hanging beside the fork, from the arm's level down past the fork plate
    parts.append({"type": "brick", "w": 1, "l": 1, "x": fork["x"] + 2, "y": fork["y"], "z": fork["z"] - 1,
                  "color": "red"})
    reqs = [q for q in r["requirements"] if q["name"] == "casters_swivel"]
    assert not checks(parts, reqs)["casters_swivel"][0]
