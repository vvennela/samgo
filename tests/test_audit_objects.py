import copy

import pytest

from samgo.audit_objects import bed_reference, bound_motion, gear_case, grid, inspect_case, rolling_reference
from samgo.eval_objects import experiment
from samgo.assembly import Sample


@pytest.mark.parametrize("span", [4, 6, 8])
def test_reference_is_valid_and_wrong_beam_sizes_fail(span):
    parts, edges = rolling_reference(span)
    case = gear_case(f"span{span}", parts, edges)
    assert inspect_case(case, case["oracle"])["passed"]
    assert not inspect_case(case, [])["passed"]
    for wrong in (4, 6, 8):
        if wrong == span:
            continue
        edits = copy.deepcopy(case["oracle"])
        for edit in edits:
            p = edit["part"]
            if p["type"] == "technic_brick":
                center = p["x"] + p["w"] // 2
                p.update(w=wrong, x=center - wrong // 2)
        assert not inspect_case(case, edits)["passed"]


def test_train_requires_all_six_axles():
    parts, edges = rolling_reference(6, axles=6, train=True)
    case = gear_case("train", parts, edges, "train")
    assert inspect_case(case, case["oracle"])["passed"]
    edits = [
        e
        for e in case["oracle"]
        if not (e["part"]["type"] in ("axle", "wheel") and e["part"]["x"] == edges[-1])
    ]
    result = inspect_case(case, edits)
    assert not result["passed"] and "missing_required_axles" in result["codes"]


def test_bed_requires_all_four_ground_supports():
    parts = bed_reference()
    case = dict(
        family="bed", mode="add", input=[p for p in parts if p["z"] >= 6], target=grid(parts).tolist()
    )
    edits = [dict(op="add", part=p) for p in parts if p["z"] < 6]
    assert inspect_case(case, edits)["passed"]
    broken = [e for e in edits if not (e["part"]["x"] == 10 and e["part"]["y"] == 6)]
    result = inspect_case(case, broken)
    assert not result["passed"] and "four_feet" in result["codes"]


def test_feedback_and_retry_share_the_same_first_answer_and_budget(tmp_path):
    parts, edges = rolling_reference(4)
    case = gear_case("span4", parts, edges)
    calls = []

    class Sampler:
        metadata = {"source": "test"}

        def __call__(self, chat, **kwargs):
            calls.append((copy.deepcopy(chat), kwargs))
            return Sample("[]", 2)

    protocol = {"max_tokens": 4000, "timeout": 60}
    record = experiment(case, Sampler(), tmp_path, protocol)
    assert len(calls) == 3 and record["feedback"]["repeated_edits"]
    assert calls[1][0][:-1] == calls[2][0][:-1]
    assert calls[1][1] == calls[2][1]
    assert calls[1][0][-1] != calls[2][0][-1]
    experiment(case, Sampler(), tmp_path, protocol)
    assert len(calls) == 3  # completed records resume without additional service calls
    with pytest.raises(ValueError, match="changed"):
        experiment(case, Sampler(), tmp_path, protocol | {"max_tokens": 2000})


def test_bound_motion_tracks_requested_side_even_when_body_sizes_tie():
    from samgo.audit_objects import brick

    parts = [
        brick(8, 2, 0, 0, 0),
        dict(type="turntable_base", w=2, l=2, x=3, y=0, z=1),
        dict(type="turntable_top", w=2, l=2, x=3, y=0, z=2),
        brick(8, 1, 0, 0, 3, "brick"),
    ]
    assert bound_motion(parts, {2, 3}, 60, "vertical")[0]
    assert not bound_motion(parts, {3}, 60, "vertical")[0]


def test_training_inventory_distinguishes_resize_from_added_beam(tmp_path):
    import json
    from samgo.audit_training import inventory

    def answer(widths):
        return json.dumps(
            [
                dict(op="add", part=dict(type="technic_brick", w=w, l=1, x=6 - w // 2, y=y, z=2))
                for y, w in enumerate(widths)
            ]
        )

    rows = [
        dict(
            messages=[
                dict(role="assistant", content=answer(before)),
                dict(role="user", content="Use a longer beam"),
                dict(role="assistant", content=answer(after)),
            ]
        )
        for before, after in [([4], [6]), ([4], [4, 6])]
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = inventory(path)
    assert result["feedback_rows_changing_beam_width_multiset"] == 2
    assert result["feedback_rows_resizing_same_center_beams"] == 1
    assert result["feedback_rows_changing_width_with_same_beam_count"] == 1
