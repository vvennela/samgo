"""Hand-made good and bad builds; S3 is done when the verifier agrees with all of them."""

import numpy as np
import pytest

from samgo.parts import COLOR_INDEX, Piece, parse_piece
from samgo.verifier import MatchConfig, verify

RED = COLOR_INDEX["red"]


def piece(w, l, type, x, y, z, color="red"):
    return {"w": w, "l": l, "type": type, "x": x, "y": y, "z": z, "color": color}


def target_of(pieces, shape):
    grid = np.zeros(shape, dtype=np.int32)
    for raw in pieces:
        p = parse_piece(raw)
        grid[p.cells] = COLOR_INDEX[p.color]
    return grid


# A 4x2 two-layer wall with staggered seams: two 2x2 bricks below, a 2x4 brick on top.
WALL = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 2, 0, 0), piece(4, 2, "brick", 0, 0, 3)]
WALL_TARGET = target_of(WALL, (4, 2, 6))


# --- valid builds -------------------------------------------------------------

def test_single_brick_on_ground():
    assert verify([piece(2, 4, "brick", 0, 0, 0)], bounds=(4, 4, 3)).passed


def test_staggered_wall_matches_target():
    report = verify(WALL, WALL_TARGET)
    assert report.passed, report.issues
    assert report.metrics["iou"] == 1.0 and report.metrics["components"] == 1


def test_rotation_is_just_swapped_size():
    assert verify([piece(4, 2, "plate", 0, 0, 0)], bounds=(4, 2, 1)).passed


def test_plate_on_brick_connects_at_height_three():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "plate", 0, 0, 3)]
    assert verify(build, bounds=(2, 2, 4)).passed


def test_one_shared_stud_is_enough():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 1, 1, 3)]
    assert verify(build, bounds=(3, 3, 6)).passed


def test_matching_shape_without_staggered_seams_fails():
    # Two 2x2s instead of one 2x4 on top: the shape matches exactly, but each
    # top brick sits on one bottom brick, leaving two separate towers.
    build = WALL[:2] + [piece(2, 2, "brick", 0, 0, 3), piece(2, 2, "brick", 2, 0, 3)]
    report = verify(build, WALL_TARGET, config=MatchConfig(min_iou=1.0))
    assert report.codes == {"disconnected"} and report.metrics["iou"] == 1.0


def test_different_parts_same_cells_pass():
    # 1x2 + 1x2 + 2x2 in place of the bottom 2x2s: different parts, same cells, still one structure.
    build = [piece(1, 2, "brick", 0, 0, 0), piece(1, 2, "brick", 1, 0, 0), piece(2, 2, "brick", 2, 0, 0), WALL[2]]
    assert verify(build, WALL_TARGET, config=MatchConfig(min_iou=1.0)).passed


# --- invalid parts ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    piece(2, 5, "brick", 0, 0, 0),             # no 2x5
    piece(3, 3, "plate", 0, 0, 0),             # no 3x3
    piece(2, 2, "tile", 0, 0, 0),              # tiles are out of the palette
    piece(2, 2, "brick", 0, 0, 0, "purple"),   # unknown color
    piece(2, 2, "brick", 0.5, 0, 0),           # off-grid
    piece(True, 2, "brick", 0, 0, 0),          # bool is not a size
    {"w": 2, "l": 2, "type": "brick"},         # missing fields
    "2x2 brick",                               # not an object
])
def test_invalid_part(bad):
    report = verify([bad], bounds=(8, 8, 8))
    assert not report.passed and report.codes == {"invalid_part"}


def test_non_list_build():
    assert verify({"pieces": []}).codes == {"invalid_part"}


def test_empty_build():
    assert verify([], bounds=(4, 4, 4)).codes == {"empty"}


# --- geometry failures ---------------------------------------------------------

@pytest.mark.parametrize("bad", [
    piece(2, 2, "brick", -1, 0, 0),
    piece(2, 2, "brick", 0, 0, -1),
    piece(2, 4, "brick", 0, 1, 0),   # sticks out past y = 4
    piece(2, 2, "brick", 0, 0, 2),   # top at z = 5 > 4
])
def test_out_of_bounds(bad):
    assert verify([bad], bounds=(4, 4, 4)).codes == {"out_of_bounds"}


def test_overlap():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 1, 0, 0)]
    report = verify(build, bounds=(4, 4, 3))
    assert "overlap" in report.codes and report.issues[0].pieces == [0, 1]


def test_plate_sunk_into_brick_overlaps():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "plate", 0, 0, 2)]
    assert "overlap" in verify(build, bounds=(2, 2, 4)).codes


def test_floating_piece():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 0, 0, 4)]  # one-plate gap
    report = verify(build, bounds=(2, 2, 7))
    assert report.codes == {"floating"} and report.issues[0].pieces == [1]


def test_floating_cluster_connected_to_itself():
    build = [piece(2, 2, "brick", 0, 0, 0),
             piece(2, 2, "brick", 4, 0, 3), piece(2, 2, "plate", 4, 0, 6)]
    report = verify(build, bounds=(6, 2, 7))
    assert report.codes == {"floating"} and report.issues[0].pieces == [1, 2]


def test_side_touching_is_not_connected():
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 2, 0, 0)]
    report = verify(build, bounds=(4, 2, 3))
    assert report.codes == {"disconnected"} and report.metrics["components"] == 2


def test_side_touching_in_the_air_is_floating():
    # The right brick touches the left tower's side but nothing holds it up.
    build = [piece(2, 2, "brick", 0, 0, 0), piece(2, 2, "brick", 0, 0, 3), piece(2, 2, "brick", 2, 0, 3)]
    assert verify(build, bounds=(4, 2, 6)).codes == {"floating"}


def test_bridge_over_gap_connects():
    build = [piece(1, 2, "brick", 0, 0, 0), piece(1, 2, "brick", 3, 0, 0), piece(4, 2, "plate", 0, 0, 3)]
    assert verify(build, bounds=(4, 2, 4)).passed


# --- target matching -----------------------------------------------------------

def test_off_target_shape():
    build = [piece(2, 2, "brick", 0, 0, 0)]
    report = verify(build, WALL_TARGET)
    assert report.codes == {"off_target"} and report.metrics["iou"] == pytest.approx(0.25)


def test_off_target_color():
    build = [dict(p, color="blue") for p in WALL]
    report = verify(build, WALL_TARGET)
    assert report.codes == {"off_target"} and report.metrics["color_accuracy"] == 0.0


def test_close_enough_passes_default_but_not_exact():
    full = [piece(2, 4, "brick", 0, 0, 0)]
    target = target_of(full, (2, 4, 3))
    target[1, 3, 2] = RED + 1  # one cell differs in color; 23/24 colors right
    assert verify(full, target).passed
    target[1, 3, 2] = 0        # one extra cell vs. target: IoU 23/24
    assert verify(full, target).passed
    assert verify(full, target, config=MatchConfig(min_iou=1.0)).codes == {"off_target"}


def test_extra_geometry_lowers_iou():
    build = WALL + [piece(2, 2, "brick", 0, 0, 0)]  # also overlaps
    assert {"overlap"} <= verify(build, WALL_TARGET).codes


def test_ldraw_id_ignores_orientation():
    assert Piece(4, 2, "brick", 0, 0, 0, "red").ldraw_id == Piece(2, 4, "brick", 0, 0, 0, "red").ldraw_id == "3001"
