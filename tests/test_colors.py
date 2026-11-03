"""Produced colors only (skipped when the Rebrickable catalogue has not been downloaded)."""

import pytest

from samgo.colors import ROOT, exists, legalize

pytestmark = pytest.mark.skipif(not (ROOT / "elements.csv.gz").exists(), reason="Rebrickable catalogue not downloaded")


def test_common_bricks_exist_and_odd_ones_do_not():
    assert exists("3001", 4)          # red 2x4 brick
    assert not exists("6578", 71)     # the 30.4x14 tyre was never made in light bluish gray


def test_wheels_get_a_black_tyre_and_a_produced_rim():
    wheel = {"type": "wheel", "axis": "y", "x": 2, "y": 0, "z": 0, "size": "30.4x14", "color": "black"}
    parts, changes = legalize([wheel])
    assert parts[0]["color"] == "black" and not changes   # the 2994 rim was made in black
    parts, changes = legalize([wheel | {"color": "dark_blue"}])
    assert parts[0]["color"] != "dark_blue" and changes and changes[0]["part"] == "2994"   # never in dark blue


def test_glass_is_split_into_produced_pieces_not_recolored():
    window = {"type": "plate", "w": 4, "l": 1, "x": 0, "y": 0, "z": 3, "color": "trans_black"}
    parts, _ = legalize([window])
    assert all(p["color"] == "trans_black" for p in parts)
    assert sum(p["w"] for p in parts) == 4 and len(parts) > 1


def test_a_body_color_that_exists_is_kept():
    brick = {"type": "brick", "w": 2, "l": 4, "x": 0, "y": 0, "z": 0, "color": "blue"}
    parts, changes = legalize([brick])
    assert parts == [brick] and not changes
