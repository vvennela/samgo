"""Validation of the research step's JSON answer (no network)."""

from samgo.research import parse_json, requirements, validate

GOOD = {
    "object": "office chair",
    "moving_parts": [
        {"name": "seat swivel", "motion": "swing", "axis": "vertical", "range_deg": 360, "count": 1,
         "lego_mechanism": "turntable 3680/3679"},
        {"name": "caster wheel", "motion": "spin", "axis": "horizontal", "range_deg": 360, "count": 5,
         "lego_mechanism": "axle + wheel"},
    ],
    "unsupported": [{"name": "gas lift", "reason": "slides vertically"}],
}


def test_good_answer_passes():
    assert validate(GOOD) == []


def test_bad_kinds_and_types_are_reported():
    bad = {"object": "", "moving_parts": [{"name": "drawer", "motion": "slide", "axis": "diagonal",
                                           "range_deg": "90", "count": 0, "lego_mechanism": ""}],
           "unsupported": ["gas lift"]}
    errors = validate(bad)
    for needle in ('"object"', "motion is 'slide'", "axis is 'diagonal'", "range_deg", "count",
                   "lego_mechanism", "unsupported[0]"):
        assert any(needle in e for e in errors), needle
    assert validate([]) == ["answer is not a JSON object"]
    assert validate({"object": "x"}) == ['"moving_parts" must be a list']
    assert any("count" in e for e in validate({"object": "x", "moving_parts": [GOOD["moving_parts"][0] | {"count": True}]}))


def test_parse_json_and_requirements():
    assert parse_json('Sure:\n```json\n{"object": "x", "moving_parts": []}\n```') == {"object": "x", "moving_parts": []}
    assert parse_json("no json here") is None
    reqs = requirements(GOOD["moving_parts"])
    assert reqs[0] == {"kind": "swing", "axis": "vertical", "degrees": 360, "count": 1, "part": "seat swivel"}
    assert reqs[1] == {"kind": "spin", "count": 5, "part": "caster wheel"}
