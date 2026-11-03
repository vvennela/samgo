import copy
import random

import numpy as np

from samgo.builder import build
from samgo.edits import apply_edits
from samgo.free_edit_test import check, explain
from samgo.parts import COLOR_INDEX
from samgo.rotor_data import KEEP, ROTOR, answer, case, feedback_for, mistakes


def block():
    grid = np.zeros((8, 8, 15), int)
    grid[:] = COLOR_INDEX["white"]
    return build(grid, attempts=1, pull=lambda p: 0.0).pieces


def verdict(body, edits):
    a = apply_edits(body, edits)
    v = check(body, a.parts, a.sources, ROTOR, stand=False)
    keep = {k: x for k, x in v.items() if k in KEEP}
    return {k: x[0] for k, x in keep.items()}, explain(body, a.parts, a.sources, a.changed, keep)


def test_rotor_on_every_face_turns_all_the_way_round():
    body = block()
    for face, F in (("+y", 8), ("-y", 0), ("+x", 8), ("-x", 0)):
        ok, _ = verdict(body, answer(face, F, 3, 6))
        assert all(ok.values()), (face, ok)


def test_loose_low_and_blocked_rotors_fail_with_feedback():
    body = block()
    good = answer("+y", 8, 3, 6)
    loose = copy.deepcopy(good)
    for e in loose:
        if e["op"] == "add" and e["part"]["type"] == "liftarm":
            e["part"].pop("axle_hole")
    ok, msgs = verdict(body, loose)
    assert not ok["rotor"] and "spin loose" in msgs[0]
    ok, msgs = verdict(body, answer("+y", 8, 3, 0))  # the arms would sweep through the table
    assert not ok["rotor"] and "below the table" in msgs[0]
    ok, msgs = verdict(body, [e for e in good if e["op"] == "add"])  # no slot cleared for the bearing
    assert not ok["structure"] and "remove" in msgs[0]


def test_generated_cases_pass_and_their_mistakes_get_feedback():
    c = next(c for c in (case(s) for s in range(8)) if c)
    a = apply_edits(c["full"], c["answer"], local=c["view"])
    v = check(c["full"], a.parts, a.sources, c["requirements"], stand=False)
    assert all(v[k][0] for k in KEEP)
    wrong = [m for m in mistakes(c, random.Random(0)) if feedback_for(c, m)]
    assert len(wrong) >= 3


def test_crop_mode_judges_new_loose_pieces_not_the_table():
    body = [p | {"z": p["z"] + 20} for p in block()]  # a piece of a bigger model, off the table
    edits = answer("+y", 8, 3, 26)
    a = apply_edits(body, edits)
    assert not check(body, a.parts, a.sources, ROTOR, stand=False)["connected"][0]
    assert check(body, a.parts, a.sources, ROTOR, stand=False, crop=True)["connected"][0]


def test_views_accept_a_model_with_a_rotor_in_it(tmp_path):
    from samgo.part_layout import render_top

    parts = apply_edits(block(), answer("+y", 8, 3, 6)).parts  # axle, bushes, liftarms: no w/l footprint
    render_top(parts, set(), tmp_path / "top.png")
    assert (tmp_path / "top.png").exists()


def test_instructions_and_ldraw_accept_a_rotor(tmp_path):
    from samgo.instructions import write
    from samgo.ldraw import export

    parts = apply_edits(block(), answer("+y", 8, 3, 6)).parts
    assert "liftarm" in write(parts, tmp_path).read_text()
    assert export(parts)[0]


def test_mechanisms_built_in_parallel_merge_and_conflicts_keep_the_first():
    from samgo.spark_mechanisms import _merge

    body = block()
    one, two = answer("+y", 8, 3, 6), answer("-y", 0, 3, 6)
    passing = [({}, {"edits": one}, ROTOR), ({}, {"edits": two}, ROTOR)]
    parts = _merge(body, passing, {})
    from samgo.kinematics import rotors
    assert rotors(parts)[0] == 2
    clash = [({}, {"edits": one}, ROTOR), (e := {}, {"edits": answer("+y", 8, 3, 6)}, ROTOR)]
    parts = _merge(body, clash, {})
    assert rotors(parts)[0] == 1 and e.get("merge_error")
