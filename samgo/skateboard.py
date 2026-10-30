"""A skateboard in real LEGO: a plate deck on two steering trucks, four spinning wheels.

Design (studs in x/y, plates in z, the board along x; LDraw numbers in brackets):

- Deck, 4 wide and `length` long: a bottom layer of 2x4 plates [3020] laid across the board, and a top layer of
  2x8 / 2x4 plates [3034 / 3020] along it, the two rows' seams staggered so every plate is held.
- Truck (one near each end), hanging from the deck by a pair of click hinge plates [44301 + 44302, 1x2] whose pin
  runs along the board at its centre line: real trucks steer by leaning about an axis along the board, and the
  click hinge is that pivot. A 1x2 plate [3023] above the deck's half (the other half has a one-plate gap above it)
  and one below the truck's half (a gap below the deck's half) keep the hinge the only link. Under it: a 2x4 plate
  [3020] across the board, a 1x2 Technic brick [3700] at each end, an axle [3707, 8 long] through both holes and a
  30.4x14 wheel [6014 + 6578] on each end, outboard of the deck.

Moves: each axle turns in its truck's round holes (4 wheels spin, all axes across the board), and each truck leans
on its hinge (horizontal axis along the board) 15 degrees each way without hitting the deck.
"""

import numpy as np

from .chair import checks
from .kinematics import BEAMS, HEIGHT, _sides, analyse, requirement_met, sweep
from .technic import verify_function

REQUIREMENTS = [
    {"name": "wheels_spin", "kind": "spin", "count": 4, "parallel": True},
    {"name": "trucks_steer", "kind": "swing", "axis": "horizontal", "degrees": 15, "count": 2},
]
UNSUPPORTED = ["truck bushings: the click hinge holds its lean by friction clicks, it does not spring back to centre",
               "steering coupling: the lean is checked, not that leaning turns the wheels' axle (real trucks' "
               "kingpin geometry is not modelled)"]


def _plate(w, l, x, y, z, color, kind="plate"):
    return {"type": kind, "w": w, "l": l, "x": x, "y": y, "z": z, "color": color}


def truck(t: int, zd: int, color: str = "light_gray") -> list[dict]:
    """A truck under the deck (deck bottom at plate zd) in columns t..t+2 of a 4-wide board (y 0..4)."""
    return [_plate(2, 1, t, 1, zd - 1, "black"),                                  # spacer above the deck half
            {"type": "click_a", "w": 2, "l": 1, "x": t, "y": 1, "z": zd - 2, "pin_x": t, "pin_y": 2,
             "pin_axis": "x", "color": "black"},
            {"type": "click_b", "w": 2, "l": 1, "x": t, "y": 2, "z": zd - 2, "pin_x": t, "pin_y": 2,
             "pin_axis": "x", "color": "black"},
            _plate(2, 1, t, 2, zd - 3, color),                                     # spacer below the truck half
            _plate(2, 4, t, 0, zd - 4, color),
            {"type": "technic_brick", "w": 2, "l": 1, "x": t, "y": 0, "z": zd - 7, "color": color, "hole": "round"},
            {"type": "technic_brick", "w": 2, "l": 1, "x": t, "y": 3, "z": zd - 7, "color": color, "hole": "round"},
            {"type": "axle", "axis": "y", "x": t + 1, "y": -2, "z": zd - 7, "length": 8},
            {"type": "wheel", "axis": "y", "x": t + 1, "y": -2, "z": zd - 7, "size": "30.4x14"},
            {"type": "wheel", "axis": "y", "x": t + 1, "y": 4, "z": zd - 7, "size": "30.4x14"}]


def build_skateboard(target: np.ndarray | None = None, length: int = 16) -> dict:
    """The skateboard's parts (shifted so every coordinate is >= 0), requirements and checks. With a target, the
    deck length is its longest horizontal extent (rounded to 12 or 16)."""
    if target is not None:
        occ = np.argwhere(np.asarray(target) > 0)
        if len(occ):
            length = int((occ.max(axis=0) - occ.min(axis=0) + 1)[:2].max())
    length = 16 if length >= 14 else 12
    zd = 7                                     # deck bottom: the trucks' Technic bricks land on z = 0
    deck = "red"
    parts = [_plate(2, 4, x, 0, zd, deck) for x in range(0, length, 2)]          # across the board
    parts += [_plate(8, 2, 0, 0, zd + 1, deck), _plate(length - 8, 2, 8, 0, zd + 1, deck),
              _plate(4, 2, 0, 2, zd + 1, deck)]                                  # seams staggered: 8 / 4 (+12)
    parts += [_plate(8, 2, 4, 2, zd + 1, deck), _plate(4, 2, 12, 2, zd + 1, deck)] if length == 16 else \
        [_plate(8, 2, 4, 2, zd + 1, deck)]
    parts += truck(2, zd) + truck(length - 4, zd)
    for p in parts:                            # wheels and axles reach to y = -2: shift the board to y >= 0
        p["y"] += 2
        if "pin_y" in p:
            p["pin_y"] += 2
    out = checks(parts, [r for r in REQUIREMENTS if r["kind"] == "spin"])
    fn = verify_function(parts)  # the full rolling-vehicle verifier: two parallel wheeled axles, wheels lowest
    out["rolls"] = (fn.passed, "; ".join(i.message for i in fn.issues) or "two parallel wheeled axles; it rolls")
    out["trucks_steer"] = steer(parts, REQUIREMENTS[1])
    return {"parts": parts, "requirements": REQUIREMENTS, "checks": out, "unsupported": UNSUPPORTED}


def steer(parts: list[dict], req: dict) -> tuple[bool, str]:
    """The horizontal-swing requirement. kinematics.requirement_met is tried first; it crashes when a horizontal
    joint's moving side has no solid part (a wheel pair spinning on its axle: sweep() stacks an empty point list),
    so on that error the same test runs here over the joints whose moving side has something solid to collide,
    and it must clear both ways (a truck leans left and right)."""
    try:
        return requirement_met(parts, {k: v for k, v in req.items() if k != "name"})
    except ValueError:
        pass
    solid = HEIGHT.keys() | BEAMS
    bodies, joints = analyse(parts)
    passed, why = 0, "no horizontal joint with a solid moving side"
    for k, (a, b, kind, point, axis, partners) in enumerate(joints):
        if abs(abs(axis[2]) - 1) < 1e-6:
            continue
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        small = min(sides, key=lambda g: sum(len(bodies[r]) for r in g))
        moving = [i for r in small for i in bodies[r]]
        if not any(parts[i]["type"] in solid for i in moving):
            continue                           # a wheel on its axle: spin, not swing
        hits = [sweep(parts, moving, point, axis, s * req["degrees"], ignore=partners) for s in (1, -1)]
        if all(h is None for h in hits):
            passed += 1
        else:
            why = f"hits something at {min(abs(h) for h in hits if h is not None):.0f} degrees"
    if passed >= req["count"]:
        return True, f"{passed} joint(s) lean {req['degrees']} degrees both ways clear (wheel axles skipped)"
    return False, f"{passed} of {req['count']} joints lean clear; {why}"


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    from .mechanism_bases import PIVOT_AS

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo/skateboard/skateboard.json")
    r = build_skateboard()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r["parts"], indent=1))
    view = [p | {"type": PIVOT_AS[p["type"]]} if p["type"] in PIVOT_AS else p for p in r["parts"]]
    out.with_name(out.stem + "_view.json").write_text(json.dumps(view))
    for k, v in r["checks"].items():
        print(k, v)
