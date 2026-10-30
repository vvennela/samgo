"""The Technic bicycle: every check passes, and each check fails when its mechanism is broken."""

import numpy as np
import pytest

from samgo.bicycle import _crank, _steering, balance, build_bicycle, max_steer, parts_for
from samgo.kinematics import analyse, connectors
from samgo.technic import TECHNIC_LDRAW, _validate


@pytest.fixture(scope="module")
def bike():
    return build_bicycle()


def test_default_bicycle_passes_every_check(bike):
    failed = {k: why for k, (ok, why) in bike["checks"].items() if not ok}
    assert not failed, failed
    assert {"structure", "connected", "running_gear", "spin", "steer", "crank", "stands"} <= bike["checks"].keys()
    assert bike["requirements"][0] == {"kind": "spin", "count": 2}
    assert bike["requirements"][1]["axis"] == "vertical" and bike["requirements"][1]["degrees"] == 30
    assert any("chain" in u for u in bike["unsupported"]) and any("brake" in u for u in bike["unsupported"])


def test_real_parts_and_five_moving_bodies(bike):
    parts = bike["parts"]
    types = {p["type"] for p in parts}
    assert {"liftarm", "pin", "axle", "wheel", "turntable_base", "turntable_top"} <= types
    assert all(p.get("friction", True) for p in parts if p["type"] == "pin")      # 2780 friction pins
    assert sum(p["type"] == "wheel" for p in parts) == 2
    bodies, joints = analyse(parts)
    # frame, fork, rear wheel, front wheel, crank
    assert len(bodies) == 5 and len(joints) == 4


def test_other_lengths_and_targets():
    for length in (14, 15, 20):
        res = build_bicycle(length=length)
        assert all(ok for ok, _ in res["checks"].values()), (length, res["checks"])
    target = np.zeros((18, 6, 20), dtype=int)
    target[:, 2:4, :10] = 1
    assert build_bicycle(target)["length"] == 18
    with pytest.raises(ValueError):
        parts_for(40)


def test_steering_has_a_limit_and_a_wheel_in_the_way_blocks_it():
    parts = parts_for()
    assert 30 <= max_steer(parts) < 90                     # the fork columns hit the head past the lock
    assert not _steering(parts, 90)[0]
    # A plate at axle height behind the front wheel: only the tyre reaches it when steering.
    assert not _steering(parts + [{"type": "plate", "w": 1, "l": 1, "x": 11, "y": 3, "z": 4, "color": "red"}], 30)[0]


def test_crank_turns_only_when_clear_and_locked():
    parts = parts_for()
    assert _crank(parts)[0]
    assert not _crank(parts + [{"type": "brick", "w": 1, "l": 1, "x": 8, "y": 0, "z": 0, "color": "red"}])[0]
    # Crank arms with round holes only would spin loose on the axle instead of turning it.
    loose = [p | {"length": 3} if "axle_hole" in p else p for p in parts]
    loose = [{k: v for k, v in p.items() if k != "axle_hole"} for p in loose]
    assert not _crank(loose)[0]


def test_without_the_kickstand_it_falls_over():
    parts = parts_for()
    assert balance(parts)[0]
    ok, why, _ = balance([p for p in parts if p.get("role") != "kickstand"])
    assert not ok and "one line" in why


def test_axle_hole_liftarm_part():
    arm = {"type": "liftarm", "axis": "x", "x": 0, "y": 0, "z": 0, "length": 2, "axle_hole": "start"}
    assert _validate(arm) is None and TECHNIC_LDRAW[("liftarm", 2)] == "60483"
    assert [c.kind for c in connectors(arm)] == ["axle_hole", "round_hole"]
    assert [c.kind for c in connectors(arm | {"axle_hole": "end"})] == ["round_hole", "axle_hole"]
    assert _validate(arm | {"length": 3}) and _validate(arm | {"axle_hole": "middle"})
    assert _validate({k: v for k, v in arm.items() if k != "axle_hole"})   # a plain 1x2 thick liftarm isn't offered
