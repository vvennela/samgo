"""A rolling office chair in real LEGO, with its mechanisms checked by the generic motion verifier.

Design (studs in x/y, plates in z; LDraw numbers in brackets):

- Base: a + shaped star of 2-wide plates [3666 2x6 / 3034 2x8] in two layers (the x arm at layer A, the y arm
  crossing over it at A+1, a 2x8 plate [3034] across
  each arm's seam at the hub, with a 2x2 plate [3022] under each y-arm end so all four casters hang at one height).
  Five arms at 72 degrees cannot sit on the stud grid, so the base has four arms (and four casters).
- Caster (one under each arm end): a 2x2 turntable [3680 base + 3679 top] hanging under the arm, the top half
  gripped by the arm, the base half carrying the fork: a 2x4 plate [3020], at each end a 1x2 brick [3004], a 1x2
  plate [3023] and a 1x2 Technic brick with a hole [3700]; an axle [32073, 5 long] through both holes carries a
  30.4x14 wheel [6014 + 6578] between the fork legs, held at both outer ends by half bushes [32123a].
- Column: 2x2 bricks [3003] on the hub, a 2x2 turntable [3680 + 3679] on top, the seat on its top half.
- Seat: two crossed layers of 2xS plates; backrest: 2xS bricks [3007 2x8] along the back edge, a 2xS plate on top.

The seat's turntable is the only link between seat+back and the base, so the seat swivels; each caster's fork turns
on its own turntable, and each wheel turns on its axle in the fork's round holes.
"""

import numpy as np

from .kinematics import _sides, analyse, connected, locked, requirement_met
from .mechanism_bases import PIVOT_AS
from .mechanism_bases import structural as pivots_as_plates
from .stability import stability
from .technic import structural as technic_structural
from .technic import verify_function
from .verifier import verify

REQUIREMENTS = [
    {"name": "casters_spin", "kind": "spin", "count": 4, "parallel": False},
    {"name": "casters_swivel", "kind": "swing", "axis": "vertical", "degrees": 360, "count": 4},
    {"name": "seat_swivels", "kind": "swing", "axis": "vertical", "degrees": 360, "count": 1},
]
UNSUPPORTED = [
    "five-arm star base: 5 arms at 72 degrees cannot sit on the stud grid; built with 4 arms and 4 casters "
    "(requirement counts lowered from 5 to 4)",
    "seat height adjustment (gas lift) and seat tilt: no LEGO-scale mechanism modelled",
]
CAR_ONLY = {"too_few_wheels", "axles_misaligned", "axle_loose"}


def _plate(w, l, x, y, z, color, kind="plate"):
    return {"type": kind, "w": w, "l": l, "x": x, "y": y, "z": z, "color": color}


def caster(x: int, y: int, z_top: int, along: str, color: str = "dark_gray") -> list[dict]:
    """A swivel caster whose 2x2 turntable occupies (x..x+2, y..y+2) with its top half at plate z_top (gripped by
    the part above). The fork runs along `along` (x or y), 4 studs long, the wheel between its legs; the Technic
    bricks' bottoms are at z_top - 9."""
    zb = z_top - 1                      # turntable base
    zf = zb - 1                         # fork plate
    zt = zf - 7                         # Technic bricks: brick (3) + plate (1) + Technic brick (3) under the fork
    parts = [{"type": "turntable_top", "w": 2, "l": 2, "x": x, "y": y, "z": z_top, "color": "black"},
             {"type": "turntable_base", "w": 2, "l": 2, "x": x, "y": y, "z": zb, "color": "black"}]
    if along == "y":                    # fork plate 2x4 along y; legs at y-1 and y+2; axle along y at edge x+1
        parts.append(_plate(2, 4, x, y - 1, zf, color))
        for ly in (y - 1, y + 2):
            parts += [_plate(2, 1, x, ly, zf - 3, color, "brick"), _plate(2, 1, x, ly, zf - 4, color),
                      {"type": "technic_brick", "w": 2, "l": 1, "x": x, "y": ly, "z": zt, "color": color,
                       "hole": "round"}]
        parts += [{"type": "axle", "axis": "y", "x": x + 1, "y": y - 1.5, "z": zt, "length": 5},
                  {"type": "wheel", "axis": "y", "x": x + 1, "y": y, "z": zt, "size": "30.4x14"},
                  {"type": "half_bush", "axis": "y", "x": x + 1, "y": y - 1.5, "z": zt, "color": "light_gray"},
                  {"type": "half_bush", "axis": "y", "x": x + 1, "y": y + 3, "z": zt, "color": "light_gray"}]
    else:                               # the same turned a quarter: fork along x, axle along x at edge y+1
        parts.append(_plate(4, 2, x - 1, y, zf, color))
        for lx in (x - 1, x + 2):
            parts += [_plate(1, 2, lx, y, zf - 3, color, "brick"), _plate(1, 2, lx, y, zf - 4, color),
                      {"type": "technic_brick", "w": 1, "l": 2, "x": lx, "y": y, "z": zt, "color": color,
                       "hole": "round"}]
        parts += [{"type": "axle", "axis": "x", "x": x - 1.5, "y": y + 1, "z": zt, "length": 5},
                  {"type": "wheel", "axis": "x", "x": x, "y": y + 1, "z": zt, "size": "30.4x14"},
                  {"type": "half_bush", "axis": "x", "x": x - 1.5, "y": y + 1, "z": zt, "color": "light_gray"},
                  {"type": "half_bush", "axis": "x", "x": x + 3, "y": y + 1, "z": zt, "color": "light_gray"}]
    return parts


def _dims(target: np.ndarray | None, size: int) -> tuple[int, int, int, int]:
    """(base span, seat size, column bricks, backrest bricks) from a target's bounding box or the defaults."""
    column, back = 3, 4
    if target is not None:
        occ = np.argwhere(np.asarray(target) > 0)
        if len(occ):
            dx, dy, dz = occ.max(axis=0) - occ.min(axis=0) + 1
            size = int(min(dx, dy))
            rest = max(int(dz) - 16, 9)       # plates above the base layers, column and seat
            column = int(np.clip(round(rest * 0.4 / 3), 1, 6))
            back = int(np.clip((rest - 3 * column) // 3, 2, 8))
    span = 16 if size >= 14 else 12           # the two arm plates meet at the hub: each is span/2 long (6 or 8)
    seat = int(np.clip(2 * round(span * 2 / 3 / 2), 4, 8))
    return span, seat, column, back


def build_chair(target: np.ndarray | None = None, size: int = 12) -> dict:
    """The chair's parts, requirements and every check (see `checks`)."""
    span, seat, n_column, n_back = _dims(target, size)
    c = span // 2 - 1                         # the hub is the 2x2 at (c..c+2, c..c+2)
    half = span // 2
    A = 10                                    # base arm layer: the casters' Technic bricks land on z = 0
    base = "black"
    parts = [_plate(half, 2, 0, c, A, base), _plate(half, 2, half, c, A, base),          # x arm
             _plate(2, 2, c, 0, A, base), _plate(2, 2, c, span - 2, A, base),             # y-arm end fillers
             _plate(2, half, c, 0, A + 1, base), _plate(2, half, c, half, A + 1, base),  # y arm, over the hub
             # Each arm's two halves meet at the hub; a 2x8 across each seam (under the x arm, over the y arm)
             # carries the arms' bending with a long lever instead of the one-stud overlap at the hub.
             _plate(8, 2, c - 3, c, A - 1, base), _plate(2, 8, c, c - 3, A + 2, base)]
    for x, y, along in ((0, c, "y"), (span - 2, c, "y"), (c, 0, "x"), (c, span - 2, "x")):
        parts += caster(x, y, A - 1, along)
    z = A + 3
    for _ in range(n_column):
        parts.append(_plate(2, 2, c, c, z, "light_gray", "brick"))
        z += 3
    parts += [{"type": "turntable_base", "w": 2, "l": 2, "x": c, "y": c, "z": z, "color": "black"},
              {"type": "turntable_top", "w": 2, "l": 2, "x": c, "y": c, "z": z + 1, "color": "black"}]
    z += 2
    s0 = c + 1 - seat // 2
    parts += [_plate(2, seat, s0 + i, s0, z, "blue") for i in range(0, seat, 2)]
    parts += [_plate(seat, 2, s0, s0 + i, z + 1, "blue") for i in range(0, seat, 2)]
    z += 2
    for _ in range(n_back):
        parts.append(_plate(seat, 2, s0, s0 + seat - 2, z, "blue", "brick"))
        z += 3
    parts.append(_plate(seat, 2, s0, s0 + seat - 2, z, "blue"))
    seat_parts = set(range(len(parts) - (seat + n_back + 1 + 1), len(parts)))  # seat turntable top onwards
    return {"parts": parts, "requirements": REQUIREMENTS, "checks": checks(parts, REQUIREMENTS, seat_parts),
            "unsupported": UNSUPPORTED}


def checks(parts: list[dict], requirements: list[dict], seat_parts: set | None = None) -> dict:
    """structure (stud rules on the grid parts, joints counted as links), connected, stands (static forces with every
    joint held, standing on the wheels' axles), wheels (the wheel mounting rules of the function verifier), each
    requirement, and for the chair that the seat swivel's moving side is exactly the seat and back."""
    solid = technic_structural(pivots_as_plates(parts))
    anchors = frozenset(k for k, p in enumerate(q for q in parts if q["type"] not in
                                                {"axle", "wheel", "bush", "pin", "axle_pin", "axle_joiner",
                                                 "half_bush", "cross_block", "liftarm"})
                        if p["type"] == "technic_brick")
    report = verify(solid, anchors=anchors)
    rule_issues = [i.message for i in report.issues if i.code not in ("floating", "disconnected")]
    out = {"structure": (not rule_issues, "; ".join(rule_issues) or "valid parts, no overlaps")}
    out["connected"] = connected(parts)
    st = stability(locked(parts, PIVOT_AS), vehicle=True)
    out["stands"] = (st.passed, "; ".join(i.message for i in st.issues) or
                     f"peak stud tension {st.metrics.get('peak_tension')} of {st.metrics.get('clutch')}")
    fn = [i for i in verify_function(parts).issues if i.code not in CAR_ONLY]
    out["wheels"] = (not fn, "; ".join(i.message for i in fn) or "axles turn in round holes, wheels held and clear")
    for req in requirements:
        out[req["name"]] = requirement_met(parts, {k: v for k, v in req.items() if k != "name"})
    if seat_parts is not None:
        out["seat_is_moving_side"] = _seat_moves(parts, seat_parts)
    return out


def _seat_moves(parts: list[dict], seat_parts: set) -> tuple[bool, str]:
    """Some vertical joint's moving side is exactly the seat and backrest (with the turntable's top half)."""
    bodies, joints = analyse(parts)
    for k, j in enumerate(joints):
        if abs(abs(j[4][2]) - 1) > 1e-6:
            continue
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        small = min(sides, key=lambda g: sum(len(bodies[r]) for r in g))
        if {i for r in small for i in bodies[r]} == seat_parts:
            return True, f"the seat and back ({len(seat_parts)} parts) turn on the column's turntable"
    return False, "no vertical joint turns exactly the seat and back"


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo/chair/chair.json")
    r = build_chair()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r["parts"], indent=1))
    view = [p | {"type": PIVOT_AS[p["type"]]} if p["type"] in PIVOT_AS else p for p in r["parts"]]
    out.with_name(out.stem + "_view.json").write_text(json.dumps(view))
    for k, v in r["checks"].items():
        print(k, v)
