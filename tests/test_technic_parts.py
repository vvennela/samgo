"""Technic connection parts: pins, axle pins, axle joiners, cross blocks, liftarms, half bushes."""

import numpy as np

from samgo.kinematics import analyse, connected, requirement_met, sweep, unseated
from samgo.technic import _validate, structural, verify_function
from samgo.verifier import verify


def beam():
    """A 1x8 Technic brick along x (holes on stud edges 1..7, across y) and a 1x5 liftarm against its -y side,
    holes on edges 1..5, lined up with the brick's."""
    return [{"type": "technic_brick", "w": 8, "l": 1, "x": 0, "y": 1, "z": 0, "color": "black", "hole": "round"},
            {"type": "liftarm", "axis": "x", "x": 1, "y": 0, "z": 0, "length": 5, "color": "light_gray"}]


def pin(x, friction=True):
    return {"type": "pin", "axis": "y", "x": x, "y": 0, "z": 0, "length": 2, "friction": friction}


def body_of(bodies, k):
    return next(r for r, members in bodies.items() if k in members)


def test_new_parts_validate_and_bad_ones_are_rejected():
    good = [pin(1), pin(1, friction=False), pin(1) | {"length": 3},
            {"type": "axle_pin", "axis": "y", "x": 1, "y": 0, "z": 0, "axle_end": "start"},
            {"type": "axle_joiner", "axis": "y", "x": 1, "y": 0, "z": 0},
            {"type": "half_bush", "axis": "x", "x": 1, "y": 0, "z": 0},
            {"type": "cross_block", "axis": "y", "x": 1, "y": 0, "z": 0, "pin_at": "-x"},
            {"type": "pin", "axis": "z", "x": 2, "y": 0.5, "z": 2.5}] + beam()[1:]
    assert all(_validate(p) is None for p in good), [_validate(p) for p in good]
    bad = [pin(1, friction=False) | {"length": 3}, beam()[1] | {"length": 4},
           {"type": "cross_block", "axis": "y", "x": 1, "y": 0, "z": 0, "pin_at": "+y"}, pin(1) | {"x": 1.3}]
    assert all(_validate(p) for p in bad)


def test_a_liftarm_pinned_with_two_friction_pins_is_connected_and_rigid():
    parts = beam() + [pin(1), pin(5)]
    bodies, joints = analyse(parts)
    assert len(bodies) == 1 and not joints
    assert connected(parts)
    assert not unseated(parts)
    # Pins and liftarms hold by pins, not studs: the stud verifier sees only the brick, and it stands.
    assert [p["type"] for p in structural(parts)] == ["brick"]
    assert verify(structural(parts)).passed


def test_two_frictionless_pins_in_two_holes_also_hold_it_rigid():
    bodies, joints = analyse(beam() + [pin(1, False), pin(3, False)])
    assert len(bodies) == 1 and not joints


def test_a_liftarm_on_one_frictionless_pin_swings_and_a_stop_limits_it():
    parts = beam() + [pin(1, friction=False)]
    bodies, joints = analyse(parts)
    assert len(joints) == 1 and joints[0][2] == "revolute"
    assert abs(abs(joints[0][4][1]) - 1) < 1e-9          # turns about the pin, across the beam (y)
    assert connected(parts)                              # hangs on its pin
    a, b, _, point, axis, partners = joints[0]
    moving = bodies[b] if 0 in bodies[a] else bodies[a]
    assert 1 in moving and 0 not in moving
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "horizontal", "degrees": 90})
    assert ok, why
    # A 1x2 brick on top of the Technic brick, reaching over the liftarm: the arm swings down freely but hits
    # the stop going up.
    stop = {"type": "brick", "w": 1, "l": 2, "x": 3, "y": 0, "z": 3, "color": "red"}
    parts = parts + [stop]
    bodies, joints = analyse(parts)
    a, b, _, point, axis, partners = joints[0]
    moving = bodies[b] if 0 in bodies[a] else bodies[a]
    hits = [sweep(parts, moving, point, axis, d, ignore=partners) for d in (90, -90)]
    assert hits.count(None) == 1
    assert 0 < abs(next(h for h in hits if h is not None)) < 90


def test_an_axle_through_a_liftarm_carries_a_spinning_wheel():
    # The liftarm is held by two friction pins; a 3-long axle runs through its free hole 4, through the brick's
    # hole 4 behind it, and out the front with a wheel on it.
    parts = beam() + [pin(1), pin(5),
                      {"type": "axle", "axis": "y", "x": 4, "y": -2, "z": 0, "length": 4, "stop": True},
                      {"type": "wheel", "axis": "y", "x": 4, "y": -2, "z": 0}]
    bodies, joints = analyse(parts)
    wheel_body = body_of(bodies, 5)
    assert 4 in bodies[wheel_body]                       # the axle and the wheel turn together
    assert any(wheel_body in (j[0], j[1]) and j[2] == "revolute" and abs(abs(j[4][1]) - 1) < 1e-9 for j in joints)
    assert requirement_met(parts, {"kind": "spin", "count": 1})[0]


def test_liftarm_holes_support_axles_in_the_function_verifier():
    # The axle at edge 4 runs through the liftarm's hole and the brick's: two aligned round holes.
    parts = beam() + [pin(1), pin(5), {"type": "axle", "axis": "y", "x": 4, "y": -2, "z": 0, "length": 4, "stop": True}]
    assert "axle_unsupported" not in verify_function(parts).codes


def test_an_axle_joiner_joins_two_axles_in_line():
    a1 = {"type": "axle", "axis": "y", "x": 0, "y": 0, "z": 0, "length": 3}
    joiner = {"type": "axle_joiner", "axis": "y", "x": 0, "y": 2, "z": 0}
    a2 = {"type": "axle", "axis": "y", "x": 0, "y": 3, "z": 0, "length": 3}
    bodies, _ = analyse([a1, joiner, a2])
    assert len(bodies) == 1
    bodies, _ = analyse([a1, a2 | {"y": 10}])
    assert len(bodies) == 2                              # without it they are two loose axles


def test_a_cross_block_holds_an_axle_and_takes_a_pin_across_it():
    # Axle along y through the block's axle hole; its pin hole one stud up, across x, pinned into a liftarm's hole
    # (a liftarm along y beside the block).
    block = {"type": "cross_block", "axis": "y", "x": 1, "y": 0, "z": 0, "pin_at": "+z"}
    axle = {"type": "axle", "axis": "y", "x": 1, "y": -1, "z": 0, "length": 3}
    arm = {"type": "liftarm", "axis": "y", "x": 1.5, "y": 0.5, "z": 2.5, "length": 3}
    p = {"type": "pin", "axis": "x", "x": 0.5, "y": 0.5, "z": 2.5, "length": 2}
    bodies, joints = analyse([block, axle, arm, p])
    assert len(bodies) == 1 and not joints
    assert not unseated([block, axle, arm, p])


def test_an_axle_pin_locks_into_an_axle_hole_and_pins_into_a_round_one():
    parts = beam() + [{"type": "axle_pin", "axis": "y", "x": 2, "y": 0, "z": 0, "axle_end": "start"},
                      {"type": "half_bush", "axis": "y", "x": 2, "y": -1, "z": 0}]
    # The axle end fills the liftarm's hole, the pin end (with friction) grips the brick's.
    bodies, joints = analyse(parts)
    assert body_of(bodies, 2) == body_of(bodies, 0) != body_of(bodies, 1)
    assert len(joints) == 1 and set(joints[0][:2]) == {body_of(bodies, 0), body_of(bodies, 1)}  # turns in the arm
    assert [k for k, _ in unseated(parts)] == [3]        # the half bush, past the axle end, grips nothing


def test_a_pin_not_in_any_hole_is_reported():
    parts = beam() + [pin(1), pin(5), pin(1) | {"z": 6}]
    loose = unseated(parts)
    assert [k for k, _ in loose] == [4]
    assert "connector_loose" in verify_function(parts).codes
    assert not connected(parts)
