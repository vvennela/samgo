"""The clear edit: empties a box in the whole model and re-lays what crosses its edge as real plates."""

from samgo.edits import apply_edits


def test_clear_cuts_a_crossing_plate_and_keeps_its_identity():
    body = [{"type": "plate", "w": 6, "l": 1, "x": 0, "y": 0, "z": 0, "color": "red"},
            {"type": "plate", "w": 6, "l": 1, "x": 0, "y": 0, "z": 1, "color": "red"}]
    a = apply_edits(body, [{"op": "clear", "box": [2, 2, 0, 0, 1, 1]}])
    top = [p for p in a.parts if p["z"] == 1]
    cells = {x for p in top for x in range(p["x"], p["x"] + p["w"])}
    assert cells == {0, 1, 3, 4, 5}                       # the box is empty, the rest still covered
    assert 1 in a.sources                                  # the biggest piece is still input part 1
    assert any(p["z"] == 0 and p["w"] == 6 for p in a.parts)  # the layer below is untouched


def test_clear_reaches_parts_outside_the_local_view():
    body = [{"type": "plate", "w": 2, "l": 2, "x": 0, "y": 0, "z": 0, "color": "red"},
            {"type": "plate", "w": 4, "l": 1, "x": 0, "y": 0, "z": 1, "color": "red"}]
    a = apply_edits(body, [{"op": "clear", "box": [3, 3, 0, 0, 1, 1]}], local=[0])
    assert all(not (p["z"] == 1 and p["x"] <= 3 < p["x"] + p["w"]) for p in a.parts)
