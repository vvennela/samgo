"""A skateboard in real LEGO: a plate deck with kicked-up nose and tail on two trucks, four spinning wheels.

Design (studs in x/y, plates in z, the board along x; LDraw numbers in brackets):

- Deck, `width` (4 or 6) wide and `length` long: a bottom layer of 2xW plates laid across the board in the deck
  color, and a top layer of black 2-wide strips along it (grip tape), the strips' seams staggered so every plate is
  held. Nose and tail kick up in two steps (plates over the last two studs, then the last one).
- Truck (one near each end), rigid: a 2x2 brick under the deck's centre, a 2xW plate across the board, a 1x2
  Technic brick [3700] at each edge, an axle through both holes (W + 4 long: 8 [3707] or 10 [3737]) and a 30.4x14
  wheel [6014 + 6578] on each end, outboard of the deck.

Moves: each axle turns in its truck's round holes (4 wheels spin, axes across the board). Steering is not modelled:
a real truck leans on a kingpin and bushings; the LDraw check showed the click-hinge version did not match the
real 44301/44302 geometry (their pivot sits past the plate ends, off the stud grid), so the trucks are rigid, as in
LEGO's own skateboard models.
"""

import numpy as np

from .chair import checks
from .kinematics import BEAMS, HEIGHT, _sides, analyse, requirement_met, sweep
from .technic import verify_function

REQUIREMENTS = [
    {"name": "wheels_spin", "kind": "spin", "count": 4, "parallel": True},
]
STEER = {"name": "trucks_steer", "kind": "swing", "axis": "horizontal", "degrees": 15, "count": 2}  # older boards
UNSUPPORTED = ["steering: real trucks lean on a kingpin and bushings; LEGO click hinges pivot off the stud grid, so "
               "these trucks are rigid (as in LEGO's skateboard models)"]


def _plate(w, l, x, y, z, color, kind="plate"):
    return {"type": kind, "w": w, "l": l, "x": x, "y": y, "z": z, "color": color}


def truck(t: int, zd: int, width: int = 4, color: str = "light_gray") -> list[dict]:
    """A rigid truck under the deck (deck bottom at plate zd) in columns t..t+2 of a `width`-wide board."""
    mid = width // 2 - 1
    return [_plate(2, 2, t, mid, zd - 3, color, "brick"),                         # column under the deck
            _plate(2, width, t, 0, zd - 4, color),                                # across the board
            {"type": "technic_brick", "w": 2, "l": 1, "x": t, "y": 0, "z": zd - 7, "color": color, "hole": "round"},
            {"type": "technic_brick", "w": 2, "l": 1, "x": t, "y": width - 1, "z": zd - 7, "color": color,
             "hole": "round"},
            {"type": "axle", "axis": "y", "x": t + 1, "y": -2, "z": zd - 7, "length": width + 4},
            {"type": "wheel", "axis": "y", "x": t + 1, "y": -2, "z": zd - 7, "size": "30.4x14"},
            {"type": "wheel", "axis": "y", "x": t + 1, "y": width, "z": zd - 7, "size": "30.4x14"}]


def _strip(length: int, y: int, z: int, offset: int, color: str) -> list[dict]:
    """2-wide plates along x covering 0..length, the first one `offset` long (staggers seams between strips)."""
    out, x = [], 0
    for n in ([offset] if offset else []) + [8] * length:
        if x >= length:
            break
        n = min(n, length - x)
        n = max((k for k in (8, 6, 4, 3, 2, 1) if k <= n), default=1)
        out.append(_plate(n, 2, x, y, z, color))
        x += n
    return out


def build_skateboard(target: np.ndarray | None = None, length: int = 30, width: int = 6) -> dict:
    """The skateboard's parts (shifted so every coordinate is >= 0), requirements and checks. With a target, the
    deck length is its longest horizontal extent."""
    if target is not None:
        occ = np.argwhere(np.asarray(target) > 0)
        if len(occ):
            length = int((occ.max(axis=0) - occ.min(axis=0) + 1)[:2].max())
    length = max(12, length - length % 2)
    width = 6 if width >= 6 else 4
    zd = 7                                     # deck bottom: the trucks' Technic bricks land on z = 0
    deck, grip = "red", "black"
    parts = [_plate(2, width, x, 0, zd, deck) for x in range(0, length, 2)]                     # across
    for k, y in enumerate(range(0, width, 2)):                                                   # grip strips
        parts += _strip(length, y, zd + 1, (4, 2, 6)[k % 3], grip)
    for x0, x1 in ((0, 2), (length - 2, length)):                                                # kick tails
        parts += [_plate(2, width, x0, 0, zd + 2, grip)]
        parts += [_plate(1, width, x0 if x0 == 0 else x1 - 1, 0, zd + 3, grip)]
    parts += truck(3, zd, width) + truck(length - 5, zd, width)
    for p in parts:                            # wheels and axles reach to y = -2: shift the board to y >= 0
        p["y"] += 2
    out = checks(parts, REQUIREMENTS)
    fn = verify_function(parts)  # the full rolling-vehicle verifier: two parallel wheeled axles, wheels lowest
    out["rolls"] = (fn.passed, "; ".join(i.message for i in fn.issues) or "two parallel wheeled axles; it rolls")
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
