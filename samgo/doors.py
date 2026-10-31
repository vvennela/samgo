"""Working hinged doors in real LEGO: car side doors and a microwave, checked by the generic motion verifier.

Hinge: the swivel hinge plate 1x4 [2429 base + 2430 top], two 1x2 halves turning about a vertical pin on their
shared edge. The base half lies in the wall in front of the door, gripped by a holder plate (1x3..1x8) of that
wall running forward from the hinge; the top half is one of the door's own layers, gripped by the door. Two hinges
on one line (the two layers farthest apart that can be held that way) carry each door, so it turns about its
front edge on the outer face of the wall.

Geometry the checker enforces (no special cases):
- the door is cut out of the wall with a plate of air above and below it, so nothing grips it but the hinges;
- a flush door that fits exactly between two walls cannot swing: its inner rear corner cuts about 1/(2L) stud into
  the wall behind it in the first few degrees (0.66 mm for a 6-long door). A car door therefore has a stud of air
  behind it (the door seam); the microwave's door hangs in front of the box with nothing beside its free end.

add_car_doors(parts, target): one door in each side wall of an arched car (samgo.arches), between the wheel
  arches, as long and tall as the wall allows; the wall around it is rebuilt by the S4 builder, the rest of the car
  is kept as it was, except for a floor pan added inside: the side walls between the arches were the car's spine,
  so plates on, in and under the deck now tie front to rear (without it the shell splits or sags).
build_microwave(size): box (floor, sides, back, roof), a front door hinged on its left edge with a dark window,
  and a 2x2 turntable [3680 + 3679] on the floor carrying a crossed plate dish.
"""

from collections import Counter

import numpy as np

from .builder import build
from .kinematics import HEIGHT, _sides, analyse, connected, locked, requirement_met, sweep
from .mechanism_bases import PIVOT_AS
from .mechanism_bases import structural as pivots_as_plates
from .parts import COLOR_INDEX, COLORS
from .stability import stability
from .technic import OFF_GRID, verify_function
from .technic import structural as technic_structural
from .verifier import verify

COLOR_NAME = {v: k for k, v in COLOR_INDEX.items()}
CAR_REQUIREMENTS = [
    {"name": "wheels_spin", "kind": "spin", "count": 4, "parallel": True},
    {"name": "doors_swing", "kind": "swing", "axis": "vertical", "degrees": 90, "count": 2},
]
MICROWAVE_REQUIREMENTS = [
    {"name": "door_swings", "kind": "swing", "axis": "vertical", "degrees": 90, "count": 1},
    {"name": "turntable_turns", "kind": "swing", "axis": "vertical", "degrees": 360, "count": 1},
]
CAR_UNSUPPORTED = ["the car gains an interior floor pan (plates on, in and under the deck) to replace the side "
                   "walls' stiffness",
                   "door seam: one stud of air behind each door (a flush door cannot swing past the wall behind it)",
                   "door latch: the door is held shut only by the hinge's friction",
                   "windows: the door stops below the window line (the shell has no glass)"]
MICROWAVE_UNSUPPORTED = ["door latch and buttons: the control side is plain plates",
                         "the turntable turns freely; no motor"]
SIZES = (8, 6, 4, 3, 2, 1)
LOOSE = 6  # loose groups of at most this many parts left by the cut are dropped; larger ones stay (and fail)


def _split(n: int, reverse: bool = False) -> list[int]:
    out = []
    while n:
        s = next(s for s in SIZES if s <= n)
        out.append(s)
        n -= s
    return out[::-1] if reverse else out


def _row(x0: int, n: int, y: int, z: int, color: str, reverse: bool = False) -> list[dict]:
    """1-wide plates along x covering x0..x0+n at row y, layer z."""
    out, x = [], x0
    for s in _split(n, reverse):
        out.append({"type": "plate", "w": s, "l": 1, "x": x, "y": y, "z": z, "color": color})
        x += s
    return out


def door(d0: int, length: int, ys: int, zb: int, zt: int, pin_y: int, color: str,
         window: tuple[int, int, int, int] | None = None, hinges: tuple[int, ...] | None = None
         ) -> tuple[list[dict], list[dict]]:
    """(hinge bases, door parts) of a door in row ys, x d0..d0+length, layers zb..zt, hinged at stud edge
    (d0, pin_y), with a hinge pair at each layer of `hinges` (default: the bottom and top layers). `window`:
    (x0, x1, z0, z1) of dark glass in the middle layers."""
    hinges = tuple(hinges or (zb, zt))
    bases, moving = [], []
    full = 0
    for z in range(zb, zt + 1):
        if z in hinges:
            bases.append({"type": "hinge_base", "w": 2, "l": 1, "x": d0 - 2, "y": ys, "z": z, "pin_x": d0,
                          "pin_y": pin_y, "color": "dark_gray"})
            moving.append({"type": "hinge_top", "w": 2, "l": 1, "x": d0, "y": ys, "z": z, "pin_x": d0,
                           "pin_y": pin_y, "color": "dark_gray"})
            moving += _row(d0 + 2, length - 2, ys, z, color)
        elif window and window[2] <= z <= window[3]:
            a, b = window[0], window[1]
            moving += _row(d0, a - d0, ys, z, color) + _row(a, b - a, ys, z, "trans_black") + \
                _row(b, d0 + length - b, ys, z, color)
        else:
            moving += _row(d0, length, ys, z, color, reverse=full % 2 == 1)
            full += 1
    return bases, moving


def _hinge_layers(occ: np.ndarray, d0: int, ys: int, zb: int, zt: int):
    """((hinge layers), holder plates) for a door: the two layers farthest apart (not adjacent) whose hinge base
    can be gripped by a plate of the wall in front running at least 3 studs forward (one layer if no pair)."""
    def holder(z):
        best = None
        for s in (1, -1):
            k = 0
            while d0 - k - 1 >= 0 and k < 8 and occ[d0 - k - 1, ys, z + s] and occ[d0 - k - 1, ys, z]:
                k += 1
            k = max((n for n in SIZES if n <= k), default=0)
            if k >= 3 and (best is None or k > best[0]):
                best = (k, z + s)
        return best
    ok = {z: h for z in range(zb, zt + 1) if (h := holder(z))}
    pairs = [(a, b) for a in ok for b in ok if b - a >= 2]
    if pairs:
        a, b = max(pairs, key=lambda q: (q[1] - q[0], ok[q[0]][0] + ok[q[1]][0]))
        layers = (a, b)
    elif ok:
        layers = (max(ok, key=lambda z: ok[z][0]),)
    else:
        return None
    holders, used = [], set()
    for z in layers:                      # two hinges may share one holder layer between them
        k, zh = ok[z]
        if zh not in used:
            used.add(zh)
            holders += _row(d0 - k, k, ys, zh, "dark_gray")
    return layers, holders


def _solid_colors(parts: list[dict]) -> list[dict]:
    """Glass colors are not in the stud verifier's palette; it sees them as black plates."""
    return [p | {"color": "black"} if p.get("color") and p["color"] not in COLORS else p for p in parts]


def joint_moves(parts: list[dict], members: set, degrees: float) -> tuple[bool, str]:
    """Some vertical joint's moving side is exactly `members`, and it turns `degrees` clear one way."""
    bodies, joints = analyse(parts)
    for k, (a, b, kind, point, axis, partners) in enumerate(joints):
        if abs(abs(axis[2]) - 1) > 1e-6:
            continue
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        small = min(sides, key=lambda g: sum(len(bodies[r]) for r in g))
        moving = [i for r in small for i in bodies[r]]
        if set(moving) != set(members):
            continue
        for sign in (1, -1):
            hit = sweep(parts, moving, point, axis, sign * degrees, steps=max(18, int(degrees / 5)),
                        ignore=partners)
            if hit is None:
                return True, f"{len(members)} parts turn {degrees} degrees clear about the pin at " \
                             f"({point[0] / 8:.0f}, {point[1] / 8:.0f}) studs"
        return False, f"its joint is found but it hits something at {abs(hit):.0f} degrees"
    return False, "no vertical joint turns exactly these parts"


def checks(parts: list[dict], requirements: list[dict], moving: dict, vehicle: bool) -> dict:
    """structure (stud rules on the grid parts, hinges as the plates they occupy; floating parts are left to
    `connected`, since a door hangs on its hinge, not on studs), connected, stands (static forces with every joint
    held), for vehicles the running gear (the full function verifier), each requirement, and for each named
    moving group that its joint moves exactly it."""
    solid = technic_structural(pivots_as_plates(_solid_colors(parts)))
    grid = [p for p in parts if p["type"] not in OFF_GRID]
    anchors = frozenset(k for k, p in enumerate(grid) if p["type"] == "technic_brick")
    report = verify(solid, anchors=anchors)
    rule_issues = [i.message for i in report.issues if i.code not in ("floating", "disconnected")]
    out = {"structure": (not rule_issues, "; ".join(rule_issues) or "valid parts, no overlaps")}
    out["connected"] = (connected(parts), "one structure through studs and joints, standing on the ground or wheels")
    st = stability(locked(parts, PIVOT_AS), vehicle=vehicle)
    out["stands"] = (st.passed, "; ".join(i.message for i in st.issues) or
                     f"peak stud tension {st.metrics.get('peak_tension')} of {st.metrics.get('clutch')}")
    if vehicle:
        fn = verify_function(pivots_as_plates(parts))
        out["running_gear"] = (fn.passed, "; ".join(i.message for i in fn.issues) or
                               "axles turn in round holes, wheels held and clear, wheels are the lowest contact")
    for req in requirements:
        out[req["name"]] = requirement_met(parts, {k: v for k, v in req.items() if k != "name"})
    for name, (members, degrees) in moving.items():
        out[name] = joint_moves(parts, members, degrees)
    return out


# ---------------------------------------------------------------------------------------------------- car doors

def _cells(p: dict):
    return (slice(int(p["x"]), int(p["x"]) + p["w"]), slice(int(p["y"]), int(p["y"]) + p["l"]),
            slice(int(p["z"]), int(p["z"]) + HEIGHT[p["type"]]))


def _overlaps(p: dict, x0, x1, y0, y1, z0, z1) -> bool:
    return p["x"] < x1 and p["x"] + p["w"] > x0 and p["y"] < y1 and p["y"] + p["l"] > y0 and \
        p["z"] < z1 and p["z"] + HEIGHT[p["type"]] > z0


def _plan(occ: np.ndarray, side: str, deck: int, front: int, rear: int):
    """Door candidates (d0, length, ys, zb, zt) on one side, best (largest, then longest) first."""
    X, Y, Z = occ.shape
    zb = deck + 2
    out = []
    for d0 in range(front + 3, rear):
        col = np.nonzero(occ[d0, :, zb])[0]
        if not len(col):
            continue
        ys = int(col.min() if side == "left" else col.max())
        outward = slice(0, ys) if side == "left" else slice(ys + 1, Y)
        for length in range(8, 2, -1):
            d1 = d0 + length
            if d1 > rear:
                continue
            for zt in range(Z - 2, zb + 1, -1):
                if not occ[d0 - 2:d1, ys, zb:zt + 1].all() or occ[d0:d1 + 1, outward, zb:zt + 1].any():
                    continue
                out.append((length * (zt - zb + 1), length, d0, ys, zb, zt))
                break
    out.sort(key=lambda c: (-c[0], -c[1], c[2]))
    return [(d0, length, ys, zb, zt) for _, length, d0, ys, zb, zt in out]


def add_car_doors(car_parts: list[dict], target: np.ndarray | None = None, attempts: int = 4) -> dict:
    """The car with a hinged door in each side wall (see the module docstring); `target` is unused beyond its
    shape check (the wall is read from the parts themselves, which is what was built)."""
    grid = [p for p in car_parts if p["type"] in HEIGHT]
    gear = [p for p in car_parts if p["type"] not in HEIGHT]
    X = max(p["x"] + p["w"] for p in grid)
    Y = max(p["y"] + p["l"] for p in grid)
    Z = max(p["z"] + HEIGHT[p["type"]] for p in grid)
    col = np.zeros((X, Y, Z), int)
    for p in grid:
        col[_cells(p)] = COLOR_INDEX.get(p.get("color"), COLOR_INDEX["light_gray"])
    occ = col > 0
    deck = min(p["z"] for p in grid if p["type"] == "technic_brick") + 3
    edges = sorted({int(round(a["x"])) for a in gear if a["type"] == "axle"})
    front, rear = edges[0], edges[-1]

    doors, removed, gaps = [], set(), []
    for side in ("left", "right"):
        chosen = None
        for d0, length, ys, zb, zt in _plan(occ, side, deck, front, rear):
            pin_y = ys if side == "left" else ys + 1
            color = COLOR_NAME[Counter(col[d0:d0 + length, ys, zb:zt + 1].ravel()).most_common(1)[0][0]]
            plan = _hinge_layers(occ, d0, ys, zb, zt)
            if plan is None:
                continue
            layers, holders = plan
            bases, moving = door(d0, length, ys, zb, zt, pin_y, color, hinges=layers)
            cut = {k for k, p in enumerate(grid) if _overlaps(p, d0 - 2, d0 + length + 1, ys, ys + 1, zb - 1, zt + 2)
                   or any(_overlaps(p, h["x"], h["x"] + h["w"], ys, ys + 1, h["z"], h["z"] + 1) for h in holders)}
            if any(grid[k]["type"] == "technic_brick" for k in cut):
                continue
            gap = [(slice(d0, d0 + length), ys, zb - 1), (slice(d0, d0 + length), ys, zt + 1),
                   (d0 + length, ys, slice(zb - 1, zt + 2))]
            # Quick sweep against everything that will stay: kept parts and 1x1 stand-ins for the cells to rebuild.
            left_over = np.zeros_like(occ)
            for k in cut:
                left_over[_cells(grid[k])] = True
            left_over[d0 - 2:d0 + length, ys, zb:zt + 1] = False
            for g in gap:
                left_over[g] = False
            stand_ins = [{"type": "plate", "w": 1, "l": 1, "x": int(x), "y": int(y), "z": int(z)}
                         for x, y, z in np.argwhere(left_over)]
            for h in holders:
                left_over[_cells(h)] = False
            trial = [p for k, p in enumerate(grid) if k not in cut] + stand_ins + holders + bases + moving
            m0 = len(trial) - len(moving)
            pin = np.array([d0 * 8.0, pin_y * 8.0, zb * 3.2])
            if not any(sweep(trial, list(range(m0, len(trial))), pin, np.array([0.0, 0, 1]), s * 90,
                             ignore=set(range(m0 - len(bases), m0))) is None for s in (1, -1)):
                continue
            chosen = (side, d0, length, ys, zb, zt, holders + bases, moving, cut, gap, layers)
            break
        if chosen is None:
            raise ValueError(f"no room for a door in the {side} wall")
        doors.append(chosen)
        removed |= chosen[8]
        gaps += chosen[9]

    kept = [p for k, p in enumerate(grid) if k not in removed]
    pivots = [q for d in doors for q in d[6] + d[7]]
    shaped = np.zeros_like(col)
    for p in kept + pivots:
        shaped[_cells(p)] = COLOR_INDEX.get(p.get("color"), COLOR_INDEX["dark_gray"])
    for k in removed:
        cells = _cells(grid[k])
        shaped[cells] = np.where(shaped[cells] > 0, shaped[cells], col[cells])
    for g in gaps:
        shaped[g] = 0
    # Floor pan: the side walls were the car's spine between front and rear; with a door cut out of each, a plate
    # layer on the deck inside the car ties front and rear together (seams only where the deck is under both sides).
    rows = sorted(d[3] for d in doors)
    pan = _floor_pan(shaped > 0, deck, rows[0] + 1, rows[1])
    for q in pan:
        shaped[_cells(q)] = COLOR_INDEX["dark_gray"]
    pivots += pan
    given = technic_structural(kept) + pivots_as_plates(pivots)

    def pull(pieces: list[dict]) -> float:
        peak = stability(pieces + gear, vehicle=True).metrics.get("peak_tension")
        return np.inf if peak is None else peak

    body = build(shaped, attempts=attempts, given=given, pull=pull, anchored=True)
    fill = body.pieces[len(given):]
    keep = _attached(kept + pivots + fill, gear)
    dropped = keep.count(False)
    nk = len(kept)
    fill = [p for p, k in zip(fill, keep[nk + len(pivots):]) if k]
    kept = [p for p, k in zip(kept, keep[:nk]) if k]
    parts = kept + pivots + fill + gear
    moving, info, n = {}, [], len(kept)
    for side, d0, length, ys, zb, zt, bases, mv, _, _, layers in doors:
        n += len(bases)
        moving[f"{side}_door_moves"] = (set(range(n, n + len(mv))), 90)
        n += len(mv)
        info.append({"side": side, "x": [d0, d0 + length], "row": ys, "z": [zb, zt], "length": length,
                     "height_plates": zt - zb + 1, "hinge_layers": list(layers)})
    return {"parts": parts, "requirements": CAR_REQUIREMENTS,
            "checks": checks(parts, CAR_REQUIREMENTS, moving, vehicle=True),
            "unsupported": CAR_UNSUPPORTED, "doors": info,
            "floor_pan": len(pan), "dropped": dropped}


def _tiling(a: int, b: int, ok) -> list[int] | None:
    """Fewest pieces (sizes from SIZES) covering a..b with every inner seam at a position where ok(seam)."""
    best = {a: []}
    todo = [a]
    while todo:
        nxt = []
        for p in todo:
            for size in SIZES:
                q = p + size
                if q > b or q in best or (q < b and not ok(q)):
                    continue
                best[q] = best[p] + [size]
                nxt.append(q)
        if b in best:
            return best[b]
        todo = nxt
    return None


def _floor_pan(taken: np.ndarray, deck: int, y0: int, y1: int) -> list[dict]:
    """A floor pan in rows y0..y1: 1-wide plates on the deck (seams only over deck on both sides), a second layer
    on them, plates filling holes in the deck under the pan, and a layer under the deck between the Technic rows
    (seams staggered): a beam up to four plates deep from the front of the car to the rear."""
    X = taken.shape[0]
    taken = taken.copy()
    mine = np.zeros_like(taken)
    pan = []

    def lay(z, where, ok_seam, touch):
        seams = set()
        for y in range(y0, y1):
            usable = [bool(where(x, y)) and not taken[x, y, z] for x in range(X)] + [False]
            start = None
            for x in range(X + 1):
                if usable[x] and start is None:
                    start = x
                if not usable[x] and start is not None:
                    sizes = _tiling(start, x, lambda s: ok_seam(s, y))
                    px = start
                    for size in sizes or []:
                        if touch(px, px + size, y):
                            pan.append({"type": "plate", "w": size, "l": 1, "x": px, "y": y, "z": z,
                                        "color": "dark_gray"})
                            taken[px:px + size, y, z] = True
                            mine[px:px + size, y, z] = True
                            seams |= {(px, y), (px + size, y)}
                        px += size
                    start = None
        return seams

    deck_at = lambda x, y: 0 <= x < X and taken[x, y, deck]  # noqa: E731
    s1 = lay(deck + 1, lambda x, y: True, lambda s, y: deck_at(s - 1, y) and deck_at(s, y),
             lambda a, b, y: taken[a:b, y, deck].any())
    on_pan = lambda x, y: mine[x, y, deck + 1]  # noqa: E731
    lay(deck + 2, on_pan, lambda s, y: (s, y) not in s1, lambda a, b, y: True)
    s0 = lay(deck, on_pan, lambda s, y: (s, y) not in s1, lambda a, b, y: True)
    # and under the deck, between the Technic rows: whatever hangs under the floor (a belly between the axles)
    # was held by the side walls; this layer grips it from above and is gripped by the deck.
    lay(deck - 1, lambda x, y: taken[x, y, deck], lambda s, y: (s, y) not in s0,
        lambda a, b, y: True)
    return pan


def _attached(grid: list[dict], gear: list[dict]) -> list[bool]:
    """For each grid part: does it hang together with the car (the group holding most parts)? Pieces held only by
    the wall that became the door are left loose by the cut and are dropped."""
    parts = grid + gear
    bodies, joints = analyse(parts)
    parent = {r: r for r in bodies}

    def find(r):
        while parent[r] != r:
            r = parent[r]
        return r
    for a, b, *_ in joints:
        parent[find(a)] = find(b)
    root = {i: find(r) for r, members in bodies.items() for i in members}
    size = Counter(root[i] for i in range(len(parts)))
    main = size.most_common(1)[0][0]
    return [root[k] == main or size[root[k]] > LOOSE for k in range(len(grid))]


# ---------------------------------------------------------------------------------------------------- microwave

def build_microwave(target: np.ndarray | None = None, size: int = 10) -> dict:
    """A microwave `size` studs wide (x), size-2 deep (y, the door row at y = 0), with a hinged front door and a
    turntable inside. With a target, the width is its longest horizontal extent (8..14)."""
    if target is not None:
        occ = np.argwhere(np.asarray(target) > 0)
        if len(occ):
            size = int((occ.max(axis=0) - occ.min(axis=0) + 1)[:2].max())
    W = int(np.clip(size, 8, 14))
    D = W - 2
    zb, zt = 2, 2 + W - 2                      # door layers; a plate of air above it, then the roof
    roof = zt + 2
    Z = roof + 2
    body = "white"
    t = np.zeros((W, D, Z), int)
    c = COLOR_INDEX[body]
    t[:, :, 0] = c                             # floor
    t[:, 1:, 1] = c
    t[0:2, 0, 1] = c
    t[:, :, roof:roof + 2] = c                 # roof
    t[0, :, zb:roof] = c                       # left side
    t[W - 1, 1:, zb:roof] = c                  # right side (the door hangs in front of its front end)
    t[:, D - 1, zb:roof] = c                   # back
    t[0:2, 0, zb:roof] = c                     # hinge pillar
    bases, moving = door(2, W - 2, 0, zb, zt, 0, "light_gray", window=(4, W - 2, zb + 2, zt - 2))
    cx, cy = W // 2 - 1, D // 2 - 1            # turntable 2x2 centred in the cavity (x 1..W-1, y 1..D-1)
    table = [{"type": "turntable_base", "w": 2, "l": 2, "x": cx, "y": cy, "z": 2, "color": "black"},
             {"type": "turntable_top", "w": 2, "l": 2, "x": cx, "y": cy, "z": 3, "color": "black"},
             {"type": "plate", "w": 4, "l": 2, "x": cx - 1, "y": cy, "z": 4, "color": "light_gray"},
             {"type": "plate", "w": 2, "l": 4, "x": cx, "y": cy - 1, "z": 5, "color": "light_gray"}]
    pivots = bases + moving + table
    for p in pivots:
        t[_cells(p)] = COLOR_INDEX["dark_gray"]
    given = pivots_as_plates(_solid_colors(pivots))
    shell = build(t, attempts=4, given=given, anchored=True).pieces[len(given):]
    parts = pivots + shell
    n = len(bases)
    groups = {"door_moves": (set(range(n, n + len(moving))), 90),
              "turntable_moves": ({n + len(moving) + 1, n + len(moving) + 2, n + len(moving) + 3}, 360)}
    out = checks(parts, MICROWAVE_REQUIREMENTS, groups, vehicle=False)
    # The door must not pass as a turntable: it stops short of a full turn.
    out["door_is_not_the_turntable"] = (not joint_moves(parts, groups["door_moves"][0], 360)[0],
                                        "the door cannot turn 360 degrees (it hits the box), so the 360 degree "
                                        "joint is the turntable")
    return {"parts": parts, "requirements": MICROWAVE_REQUIREMENTS, "checks": out,
            "unsupported": MICROWAVE_UNSUPPORTED}


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    out_dir = Path("runs/demo/doors")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if len(sys.argv) < 2 or "car" in sys.argv[1:]:
        from .arches import build_arched_car
        from .voxelize import load_mesh, voxelize

        for name in ("camry", "beetle"):
            car = build_arched_car(voxelize(load_mesh(f"runs/s6/{name}/object.glb"), 20), attempts=6)
            results[f"{name}_doors"] = add_car_doors(car.parts, car.target)
    if len(sys.argv) < 2 or "microwave" in sys.argv[1:]:
        results["microwave"] = build_microwave()
    for name, r in results.items():
        (out_dir / f"{name}.json").write_text(json.dumps(r["parts"], indent=1))
        print(name, r.get("doors", ""))
        for k, v in r["checks"].items():
            print("  ", k, v)
