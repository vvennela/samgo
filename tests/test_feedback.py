import numpy as np

from samgo.feedback import describe, retry_message
from samgo.parts import COLOR_INDEX
from samgo.tasks import EXACT
from samgo.verifier import verify

RED = COLOR_INDEX["red"]


def brick(x, y, z, w=2, l=2, color="red", type="brick"):
    return {"w": w, "l": l, "type": type, "x": x, "y": y, "z": z, "color": color}


def check(given, added, target):
    return describe(verify(given + added, target, config=EXACT), given, added, target)


def test_passing_build_gives_no_feedback():
    target = np.full((2, 2, 3), RED)
    assert check([], [brick(0, 0, 0)], target) == ""


def test_floating_piece_is_named_by_its_answer_number():
    target = np.zeros((4, 2, 4), dtype=np.int32)
    target[0:2, :, 0:3] = RED
    target[2:4, :, 3] = RED
    given = [brick(0, 0, 0)]
    added = [brick(2, 0, 3, type="plate")]
    text = check(given, added, target)
    assert "Not supported" in text and "#1 2x2 plate at (2,0,3) [overhang" in text


def test_floating_piece_one_layer_too_high_shows_the_real_top():
    target = np.zeros((2, 2, 5), dtype=np.int32)
    target[:, :, 0:4] = RED
    added = [brick(0, 0, 0), brick(0, 0, 4, type="plate")]  # brick tops out at z=3, plate placed at z=4
    target[:, :, 4] = RED
    assert "target cells under it at z=3 are still empty: (0,0) (0,1) (1,0) (1,1)" in check([], added, target)


def test_missing_cells_are_listed_by_layer():
    target = np.full((2, 4, 3), RED)
    text = check([], [brick(0, 0, 0)], target)
    assert "Target cells left empty" in text and "z=0:" in text and "(0,2)" in text


def test_wrong_color_names_the_needed_color():
    target = np.full((2, 2, 3), RED)
    text = check([], [brick(0, 0, 0, color="blue")], target)
    assert "Wrong color" in text and "needs red" in text


def test_invalid_part_explains_the_palette():
    target = np.full((2, 2, 3), RED)
    text = check([], [brick(0, 0, 0, w=5, l=5)], target)
    assert "Not a valid part" in text and "#1" in text


def test_empty_answer_and_retry_wording():
    target = np.full((2, 2, 3), RED)
    assert "no pieces" in describe(verify([], target, config=EXACT), [], [], target)
    assert "complete corrected list" in retry_message("- x")


def test_floating_piece_over_unbuilt_target_cells_names_them():
    target = np.full((2, 2, 4), RED)
    added = [brick(0, 0, 0, w=1), brick(1, 0, 3, w=1, type="plate")]  # plate over the x=1 column, never built
    assert "target cells under it at z=2 are still empty: (1,0) (1,1)" in check([], added, target)
