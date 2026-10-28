import copy

from samgo.technic import structural, verify_function
from samgo.verifier import verify


def car():
    """8 studs long (x), 2 wide; axles across y at stud edges 1 and 7, wheels on both sides."""
    chassis = [{"type": "technic_brick", "w": 8, "l": 1, "x": 0, "y": y, "z": 0, "color": "black", "hole": "round"}
               for y in (1, 2)]
    body = [{"type": "plate", "w": 8, "l": 2, "x": 0, "y": 1, "z": 3, "color": "red"},
            {"type": "brick", "w": 2, "l": 4, "x": 3, "y": 1, "z": 4, "color": "red"}]
    axles = [{"type": "axle", "axis": "y", "x": x, "z": 0, "y": -1, "length": 6} for x in (1, 7)]
    wheels = [{"type": "wheel", "axis": "y", "x": x, "z": 0, "y": y, "width": 2} for x in (1, 7) for y in (-1, 3)]
    return chassis + body + axles + wheels


def codes(parts):
    return verify_function(parts).codes


def test_reference_car_rolls_and_is_structurally_sound():
    parts = car()
    assert verify_function(parts).passed, verify_function(parts).issues
    assert verify(structural(parts)).passed


def test_axle_through_an_axle_shaped_hole_is_locked():
    parts = car()
    parts[0]["hole"] = "axle"
    assert "axle_locked" in codes(parts)


def test_part_in_the_axle_path_blocks_it():
    parts = car() + [{"type": "brick", "w": 1, "l": 1, "x": 0, "y": 0, "z": 0, "color": "red"}]
    assert "axle_blocked" in codes(parts)


def test_body_over_the_wheel_blocks_its_sweep():
    parts = car() + [{"type": "plate", "w": 8, "l": 2, "x": 0, "y": -1, "z": 6, "color": "red"}]
    assert codes(parts) == {"wheel_blocked"}


def test_body_lower_than_the_wheels_drags():
    parts = car()
    for p in parts:
        if p["type"] in ("technic_brick", "axle", "wheel"):
            p["z"] = 3
        elif p["type"] in ("plate", "brick"):
            p["z"] += 3
    parts.append({"type": "brick", "w": 2, "l": 2, "x": 3, "y": 1, "z": 0, "color": "red"})
    assert "body_drags" in codes(parts)


def test_axle_at_the_wrong_height_misses_the_holes():
    parts = car()
    parts[4]["z"] = 3
    assert "axle_unsupported" in codes(parts)


def test_one_axle_is_not_a_car_and_wheels_must_sit_on_axles():
    parts = [p for p in car() if not (p["type"] in ("axle", "wheel") and p["x"] == 7)]
    assert "too_few_wheels" in codes(parts)
    parts = car()
    parts[-1]["x"] = 5
    assert "wheel_off_axle" in codes(parts)


def test_bad_parts_are_rejected():
    parts = car()
    parts[0] = parts[0] | {"w": 3}
    assert codes(parts) == {"invalid_part"}


def test_axles_that_are_not_parallel_cannot_roll():
    front = [p for p in car() if not (p["type"] in ("axle", "wheel") and p["x"] == 7)]
    # A second assembly turned 90 degrees: Technic bricks long in y, an axle along x, wheels on it.
    turned = [{"type": "technic_brick", "w": 1, "l": 8, "x": x, "y": 0, "z": 0, "color": "black", "hole": "round"}
              for x in (20, 21)]
    turned += [{"type": "axle", "axis": "x", "y": 1, "z": 0, "x": 18, "length": 6}]
    turned += [{"type": "wheel", "axis": "x", "y": 1, "z": 0, "x": x, "width": 2} for x in (18, 22)]
    assert "axles_misaligned" in codes(front + turned)
