"""The skateboard: plate deck, two trucks leaning on click hinges, four spinning wheels."""

from samgo.skateboard import build_skateboard, steer


def ok(v):
    return v if isinstance(v, bool) else v[0]


def test_default_skateboard_passes_every_check():
    r = build_skateboard()
    failed = {k: v for k, v in r["checks"].items() if not ok(v)}
    assert not failed, failed
    for k in ("structure", "connected", "stands", "wheels_spin", "rolls", "trucks_steer"):
        assert k in r["checks"]


def test_skateboard_parts():
    parts = build_skateboard()["parts"]
    n = lambda t: sum(p["type"] == t for p in parts)  # noqa: E731
    assert n("wheel") == 4 and n("axle") == 2 and n("click_a") == 2 and n("click_b") == 2
    assert min(p["y"] for p in parts) >= 0 and min(p["z"] for p in parts) >= 0


def test_a_truck_cannot_lean_too_far():
    parts = build_skateboard()["parts"]
    assert steer(parts, {"kind": "swing", "axis": "horizontal", "degrees": 15, "count": 2})[0]
    assert not steer(parts, {"kind": "swing", "axis": "horizontal", "degrees": 60, "count": 2})[0]


def test_short_board_from_a_target_passes():
    import numpy as np

    r = build_skateboard(np.ones((12, 4, 6), dtype=int))
    assert all(ok(v) for v in r["checks"].values()), r["checks"]
