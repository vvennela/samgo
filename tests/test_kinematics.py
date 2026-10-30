"""The connector experiment: one generic analysis, four builds, no mechanism-specific code."""

import numpy as np

from samgo.kinematics import analyse, sweep


def car(locked=False):
    chassis = [{"type": "technic_brick", "w": 8, "l": 1, "x": 0, "y": y, "z": 0, "color": "black",
                "hole": "axle" if (locked and y == 1) else "round"} for y in (1, 2)]
    body = [{"type": "plate", "w": 8, "l": 2, "x": 0, "y": 1, "z": 3, "color": "red"}]
    gear = [{"type": "axle", "axis": "y", "x": x, "z": 0, "y": -1, "length": 6} for x in (1, 7)]
    gear += [{"type": "wheel", "axis": "y", "x": x, "z": 0, "y": y} for x in (1, 7) for y in (-1, 3)]
    return chassis + body + gear


def door(wall=False):
    """A floor, a swivel hinge (base on the floor, top under the door), and a 1x4 door standing on the top half.
    The pin is vertical at stud corner (2, 1)."""
    # The floor is under the hinge's base half only: the top half (and the door on it) is held by the pin alone.
    parts = [{"type": "plate", "w": 2, "l": 6, "x": 0, "y": 0, "z": 0, "color": "gray"},
             {"type": "hinge_base", "w": 2, "l": 1, "x": 0, "y": 0, "z": 1, "pin_x": 2, "pin_y": 1, "color": "gray"},
             {"type": "hinge_top", "w": 2, "l": 1, "x": 2, "y": 0, "z": 1, "pin_x": 2, "pin_y": 1, "color": "gray"},
             {"type": "brick", "w": 4, "l": 1, "x": 2, "y": 0, "z": 2, "color": "red"}]
    if wall:  # a post in the door's swing path
        parts.append({"type": "brick", "w": 1, "l": 1, "x": 3, "y": 2, "z": 2, "color": "blue"})
    return parts


def moving_body(parts, bodies, joint):
    """The body on the joint that does not contain part 0 (the floor or chassis)."""
    a, b = joint[0], joint[1]
    return bodies[b] if 0 in bodies[a] else bodies[a]


def test_car_wheels_are_found_as_spinning_bodies():
    bodies, joints = analyse(car())
    assert len(joints) >= 2 and all(j[2] == "revolute" for j in joints)
    assert all(abs(abs(j[4][1]) - 1) < 1e-9 for j in joints)  # every spin axis runs across the car (y)


def test_an_axle_in_an_axle_hole_is_locked():
    free = len(analyse(car())[1])
    locked = len(analyse(car(locked=True))[1])
    assert locked < free


def test_a_door_on_a_swivel_hinge_swings_90_degrees_clear():
    parts = door()
    bodies, joints = analyse(parts)
    assert len(joints) == 1 and abs(joints[0][4][2]) == 1  # one joint, vertical axis
    moving = moving_body(parts, bodies, joints[0])
    assert 3 in moving
    # The door swings off the floor's edge (toward +y); the hinge's own halves turn past each other.
    assert sweep(parts, moving, joints[0][3], joints[0][4], 90, ignore=joints[0][5]) is None


def test_a_post_in_the_doors_path_stops_the_swing():
    parts = door(wall=True)
    bodies, joints = analyse(parts)
    moving = moving_body(parts, bodies, joints[0])
    hit = sweep(parts, moving, joints[0][3], joints[0][4], 90, ignore=joints[0][5])
    assert hit is not None and hit <= 90


from samgo.kinematics import requirement_met  # noqa: E402


def test_requirement_spin_passes_for_the_car_and_fails_when_a_wheel_is_locked():
    assert requirement_met(car(), {"kind": "spin", "count": 4})[0]
    ok, why = requirement_met(car(locked=True), {"kind": "spin", "count": 4})
    assert not ok and "can spin" in why


def test_a_preset_car_door_turns_on_its_stud():
    # A 6x2 floor; a 1x3x1 car door standing on the floor's corner stud (1, 0), hinged there, held by a plate on top.
    parts = [{"type": "plate", "w": 2, "l": 6, "x": 0, "y": 0, "z": 0, "color": "gray"},
             {"type": "car_door", "w": 1, "l": 3, "x": 1, "y": 0, "z": 1, "pin_x": 1, "pin_y": 0, "color": "red"}]
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "vertical", "degrees": 90})
    assert ok, why


def test_a_laptop_lid_on_click_hinges_swings_about_a_horizontal_axis():
    # Base 4x4 plate; click hinge halves along its back edge (pin along x at y=4); the lid is a 4x4 plate on top half,
    # standing up from the hinge, held only by the pin.
    parts = [{"type": "plate", "w": 4, "l": 4, "x": 0, "y": 0, "z": 0, "color": "gray"},
             {"type": "click_a", "w": 4, "l": 1, "x": 0, "y": 3, "z": 1, "pin_x": 0, "pin_y": 4, "pin_axis": "x", "color": "gray"},
             {"type": "click_b", "w": 4, "l": 1, "x": 0, "y": 4, "z": 1, "pin_x": 0, "pin_y": 4, "pin_axis": "x", "color": "gray"},
             {"type": "brick", "w": 4, "l": 1, "x": 0, "y": 4, "z": 2, "color": "black"}]
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "horizontal", "degrees": 60})
    assert ok, why


def test_a_tv_on_a_turntable_turns():
    parts = [{"type": "plate", "w": 4, "l": 4, "x": 0, "y": 0, "z": 0, "color": "gray"},
             {"type": "turntable_base", "w": 2, "l": 2, "x": 1, "y": 1, "z": 1, "color": "gray"},
             {"type": "turntable_top", "w": 2, "l": 2, "x": 1, "y": 1, "z": 2, "color": "gray"},
             {"type": "brick", "w": 6, "l": 1, "x": -1, "y": 1, "z": 3, "color": "black"}]
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "vertical", "degrees": 60})
    assert ok, why
