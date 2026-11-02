import random

from samgo.edits import apply_edits
from samgo.flap_data import feedback_for, mistakes, sheet_body
from samgo.flap_sites import FLAP, pick, site_answer, sites
from samgo.free_edit_test import check
from samgo.part_layout import handoff, parse


def slab():
    """A small generated body (block + thin structure, flap edge +x) that has flap sites."""
    parts, out = sheet_body(random.Random(4))
    return parts, sorted(out)


def test_sites_are_verified_and_hinges_sit_on_top():
    parts, wing = slab()
    from samgo.primitives import occupancy

    found = sites(parts, wing)
    assert found
    occ = occupancy(parts)
    for s in found[:3]:
        a = apply_edits(parts, site_answer(parts, s))
        v = check(parts, a.parts, a.sources, FLAP, stand=False)
        assert v["structure"][0] and v["connected"][0]
        for y in s["hinge_rows"]:  # on top: the hinge cells are empty, the cells under them are not
            assert all((x, y, s["z"]) not in occ for x in range(s["xh"] - 2, s["xh"] + 3))
            assert any((x, y, s["z"] - 1) in occ for x in range(s["xh"] - 2, s["xh"] + 3))
    assert pick(found, near_y=7) in found


def test_hinge_inside_the_top_layer_is_caught():
    parts, wing = slab()
    s = sites(parts, wing)[0]
    edits = [e if e["op"] != "add" else {"op": "add", "part": e["part"] | {"z": s["z"] - 1}}
             for e in site_answer(parts, s)]
    a = apply_edits(parts, edits)
    assert not check(parts, a.parts, a.sources, FLAP, stand=False)["structure"][0]


def test_wing_bodies_and_mistakes():
    from samgo.flap_data import case

    c = next(c for c in (case(s) for s in range(6)) if c)
    wrong = [m for m in mistakes(c, random.Random(0)) if feedback_for(c, m)]
    assert wrong and all(m != c["answer"] for m in wrong)
    parts, out = sheet_body(random.Random(4))
    assert out and all(p["type"] == "plate" for p in parts)


def test_spark_answer_parsing_and_handoff_on_any_edge():
    assert parse('{"edge": "+x", "from": 9, "to": 4}') == {"edge": "+x", "from": 4, "to": 9}
    assert parse('{"edge": "up"}') is None
    parts, wing = slab()
    s0 = sites(parts, wing)[0]
    lo, hi = s0["y0"], s0["y1"]
    h = handoff(parts, wing, {"edge": "+x", "from": lo, "to": hi})
    assert h and h["axis"] == "x" and h["rows"][0] <= hi and h["rows"][1] >= lo
    from samgo.primitives import swap_xy

    turned = [swap_xy(p) for p in parts]  # the same wing with its trailing edge facing +y
    h2 = handoff(turned, wing, {"edge": "+y", "from": lo, "to": hi})
    assert h2 and h2["axis"] == "y" and h2["site"] == h["site"]
