"""Hinged doors: the photo cars get a working door in each side, the microwave a door and a turntable, every check
passes, and something in a door's path makes it fail."""

import pytest

from samgo.doors import CAR_REQUIREMENTS, add_car_doors, build_microwave, checks, joint_moves


def ok(v):
    return v if isinstance(v, bool) else v[0]


@pytest.fixture(scope="module")
def cars():
    from samgo.arches import build_arched_car
    from samgo.voxelize import load_mesh, voxelize

    out = {}
    for name in ("camry", "beetle"):
        car = build_arched_car(voxelize(load_mesh(f"runs/s6/{name}/object.glb"), 20), attempts=6)
        out[name] = add_car_doors(car.parts, car.target)
    return out


@pytest.mark.parametrize("name", ["camry", "beetle"])
def test_photo_car_doors_pass_every_check(cars, name):
    r = cars[name]
    failed = {k: v for k, v in r["checks"].items() if not ok(v)}
    assert not failed, failed
    for k in ("structure", "connected", "stands", "running_gear", "wheels_spin", "doors_swing",
              "left_door_moves", "right_door_moves"):
        assert k in r["checks"]
    assert len(r["doors"]) == 2 and all(d["length"] >= 5 and d["height_plates"] >= 3 for d in r["doors"])
    n = lambda t: sum(p["type"] == t for p in r["parts"])  # noqa: E731
    assert n("hinge_base") == n("hinge_top") == 4 and n("wheel") == 4


def test_a_brick_in_the_doors_path_stops_it(cars):
    r = cars["camry"]
    left = next(d for d in r["doors"] if d["side"] == "left")
    parts = list(r["parts"])
    d0, row, (zb, zt) = left["x"][0], left["row"], left["z"]
    # A brick standing outside the car, 2 studs from the hinge, inside the arc the door's free end sweeps.
    parts.append({"type": "brick", "w": 2, "l": 1, "x": d0 + 1, "y": row - 3, "z": zb, "color": "red"})
    reqs = [q for q in CAR_REQUIREMENTS if q["name"] == "doors_swing"]
    out = checks(parts, reqs, {}, vehicle=True)
    assert not out["doors_swing"][0]
    assert "1 of 2" in out["doors_swing"][1]


def test_microwave_door_and_turntable():
    r = build_microwave()
    failed = {k: v for k, v in r["checks"].items() if not ok(v)}
    assert not failed, failed
    for k in ("door_swings", "turntable_turns", "door_moves", "turntable_moves", "door_is_not_the_turntable"):
        assert k in r["checks"]
    parts = r["parts"]
    assert sum(p["type"] == "turntable_base" for p in parts) == 1
    assert any(p.get("color") == "trans_black" for p in parts)
    # the turntable's side is the 3 parts on its top half; the door's side is the door
    dish = {i for i, p in enumerate(parts) if p["type"] == "turntable_top"}
    dish |= {i for i, p in enumerate(parts) if p["type"] == "plate" and p["z"] in (4, 5) and p["color"] == "light_gray"
             and 1 <= p["y"]}
    assert joint_moves(parts, dish, 360)[0]


def test_microwave_door_blocked_by_a_post_fails():
    r = build_microwave()
    # a post hanging from a plate on the roof, in front of the door (attached, so it stays put)
    parts = list(r["parts"]) + [{"type": "plate", "w": 1, "l": 3, "x": 5, "y": -2, "z": 14, "color": "red"},
                                {"type": "brick", "w": 1, "l": 1, "x": 5, "y": -2, "z": 11, "color": "red"},
                                {"type": "brick", "w": 1, "l": 1, "x": 5, "y": -2, "z": 8, "color": "red"}]
    door = {i for i, p in enumerate(r["parts"]) if p["y"] == 0 and p["x"] >= 2 and 2 <= p["z"] <= 10}
    assert joint_moves(r["parts"], door, 90)[0]
    blocked = joint_moves(parts, door, 90)
    assert not blocked[0] and "hits" in blocked[1], blocked
