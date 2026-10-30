"""A LEGO Technic bicycle with working mechanisms, checked by the generic verifiers.

Layout (x along the bike, rear wheel at small x; y across it; z up). The road wheels (30.4x14: 6578 tyre on
6014 rim) sit in line on stop axles, each between two thick liftarms:

- rear stays: two 1x7 liftarms (32524) either side of the rear wheel, the rear axle through their first hole;
- bottom bracket: two 1x6 (or 1x4) Technic bricks between the stays, pinned to them with friction pins (2780);
  the crank axle runs through the stays and these bricks, a crank arm (60483, axle hole) locked on each end,
  pointing opposite ways;
- seat tube: bricks stacked on the bottom bracket up to a pair of 1x4 Technic bricks, a seat post and saddle;
- top tube: two liftarms pinned between the seat-tube Technic bricks and the head Technic bricks;
- head: a 2x2 turntable (3680 base on the head bricks, 3679 top) with the steering axis vertical, straight
  above the front axle; the fork's crown plate sits on the turntable top;
- fork: from the crown, two brick columns down either side of the front wheel to 1x4 Technic bricks, each
  pinned to a 1x3 liftarm (32523) that carries the front axle;
- handlebar: a 1x9 liftarm (40490) across the bike, pinned to a 1x6 Technic brick on the crown;
- double kickstand: a leg each side (a 1x4 Technic brick pinned to the stay, a 1x4 brick under it to the
  ground), so the bike stands on two wheels and two feet.

Checks: part validity and the stud rules (S3a), one connected structure with every pin seated, the running gear
rules (verify_function, minus the car-only "two wheels per axle" and "body is not the lowest point" rules), the
research requirements through kinematics.requirement_met, the crank turning a full circle (analyse + sweep), the
steered front wheel clearing the frame, and static balance: the centre of mass over the ground contacts.
"""

import json
from pathlib import Path

import numpy as np

from .kinematics import _sides, analyse, box, connected, requirement_met, sweep, unseated
from .mechanism_bases import PIVOT_AS
from .technic import (DEFAULT_WHEEL, HOLE_HEIGHT, PLATE, STUD, WHEELS, _validate, structural,
                      verify_function, wheel_studs)
from .verifier import verify

WHEEL = "30.4x14"
REQUIREMENTS = [
    {"kind": "spin", "count": 2},
    {"kind": "swing", "axis": "vertical", "degrees": 30, "count": 1},
]
UNSUPPORTED = ["chain (no chain links in the part set; the crank turns on its own)",
               "gears / sprockets (no gears in the part set)",
               "brakes", "pedals (the crank arms' free pin holes take none)"]
LENGTHS = range(14, 21)
BALANCE_MARGIN = 4.0  # mm: the centre of mass must lie this far inside the ground contacts


def _pin(x, y, z, axis="y"):
    return {"type": "pin", "axis": axis, "x": x, "y": y, "z": z, "length": 2, "friction": True, "color": "black"}


def _length_from(target: np.ndarray) -> int:
    xs, ys = np.nonzero((target > 0).any(axis=2))
    span = max(int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1) if len(xs) else 16
    return int(min(max(span, LENGTHS[0]), LENGTHS[-1]))


def parts_for(length: int = 16) -> list[dict]:
    if length not in LENGTHS:
        raise ValueError(f"bicycle length must be {LENGTHS[0]}..{LENGTHS[-1]} studs")
    Y = 2                                   # the wheels' row: they span y Y..Y+2
    zA = 3                                  # axle plate height: a 30.4 wheel's bottom is then on the ground
    XR, XF, XC = 2, length - 2, 7           # rear axle, front axle (= steering axis), crank axle
    L, R = Y - 1, Y + 2                     # liftarm rows either side of the wheels
    OL, OR = Y - 2, Y + 3                   # outer rows: fork columns, crank arms, kickstand legs
    frame, fork, gear = "red", "black", "light_gray"
    parts: list[dict] = []
    add = parts.extend

    # Rear stays and the bottom bracket between them.
    inner_w = 6 if XF - 2.3 >= 10 else 4   # must stay behind the steered front wheel
    add([{"type": "liftarm", "axis": "x", "x": XR, "y": r, "z": zA, "length": 7, "color": frame} for r in (L, R)])
    add([{"type": "technic_brick", "w": inner_w, "l": 1, "x": 4, "y": r, "z": zA, "color": frame, "hole": "round"}
         for r in (Y, Y + 1)])
    for x in (5, 8 if inner_w == 6 else 6):
        add([_pin(x, L, zA), _pin(x, Y + 1, zA)])
    # Rear wheel.
    add([{"type": "axle", "axis": "y", "x": XR, "y": L, "z": zA, "length": 4, "stop": True, "color": gear},
         {"type": "wheel", "axis": "y", "x": XR, "y": Y, "z": zA, "size": WHEEL, "color": "black"}])
    # Crank: an axle through the stays and the bottom bracket, a 60483 arm locked on each end, opposite ways.
    add([{"type": "axle", "axis": "y", "x": XC, "y": OL, "z": zA, "length": 6, "color": gear},
         {"type": "liftarm", "axis": "x", "x": XC, "y": OL, "z": zA, "length": 2, "axle_hole": "start",
          "color": "dark_gray"},
         {"type": "liftarm", "axis": "x", "x": XC - 1, "y": OR, "z": zA, "length": 2, "axle_hole": "end",
          "color": "dark_gray"}])
    # Double kickstand: a leg each side, pinned to the stays behind the crank's reach.
    stand = [{"type": "technic_brick", "w": 4, "l": 1, "x": 1, "y": r, "z": zA, "color": "dark_gray",
              "hole": "round"} for r in (OL, OR)]
    stand += [{"type": "brick", "w": 4, "l": 1, "x": 1, "y": r, "z": 0, "color": "dark_gray"} for r in (OL, OR)]
    stand += [_pin(x, r, zA) for x in (3, 4) for r in (OL, R)]
    add([p | {"role": "kickstand"} for p in stand])
    # Seat tube, seat post and saddle.
    add([{"type": "brick", "w": 2, "l": 2, "x": 4, "y": Y, "z": 6, "color": frame},
         {"type": "plate", "w": 2, "l": 2, "x": 4, "y": Y, "z": 9, "color": frame}])
    add([{"type": "technic_brick", "w": 4, "l": 1, "x": 3, "y": r, "z": 10, "color": frame, "hole": "round"}
         for r in (Y, Y + 1)])
    add([{"type": "brick", "w": 2, "l": 2, "x": 4, "y": Y, "z": 13, "color": "dark_gray"},
         {"type": "plate", "w": 4, "l": 2, "x": 3, "y": Y, "z": 16, "color": "black"}])
    # Head: Technic bricks from the steering axis back, and the top tube liftarms between them and the seat tube.
    hw = next(w for w in (8, 6, 4) if XF + 1 - w >= 7)                # behind the axis, clear of the seat tube
    hx = XF + 1 - hw
    add([{"type": "technic_brick", "w": hw, "l": 1, "x": hx, "y": r, "z": 10, "color": frame, "hole": "round"}
         for r in (Y, Y + 1)])
    # The longest top tube starting in a seat-brick hole that reaches a head-brick hole and ends clear of the
    # fork's swing.
    t0, top = max(((a, n) for a in (4, 5) for n in (5, 7, 9) if hx + 1 <= a + n - 1 <= XF - 4),
                  key=lambda an: (an[1], -an[0]))
    add([{"type": "liftarm", "axis": "x", "x": t0, "y": r, "z": 10, "length": top, "color": frame} for r in (L, R)])
    head_holes = [x for x in range(hx + 1, hx + hw) if x <= t0 + top - 1]
    for x in [t0 + 1] + head_holes[-2:]:
        add([_pin(x, L, 10), _pin(x, Y + 1, 10)])
    # Steering: turntable, crown, fork columns, fork liftarms, front wheel.
    add([{"type": "turntable_base", "w": 2, "l": 2, "x": XF - 1, "y": Y, "z": 13, "color": "dark_gray"},
         {"type": "turntable_top", "w": 2, "l": 2, "x": XF - 1, "y": Y, "z": 14, "color": "light_gray"},
         {"type": "plate", "w": 2, "l": 6, "x": XF - 1, "y": OL, "z": 15, "color": fork}])
    for r in (OL, OR):
        add([{"type": "brick", "w": 2, "l": 1, "x": XF - 1, "y": r, "z": z, "color": fork} for z in (12, 9, 6)])
        add([{"type": "technic_brick", "w": 4, "l": 1, "x": XF - 2, "y": r, "z": zA, "color": fork, "hole": "round"}])
    add([{"type": "liftarm", "axis": "x", "x": XF - 1, "y": r, "z": zA, "length": 3, "color": fork} for r in (L, R)])
    for x in (XF - 1, XF + 1):
        add([_pin(x, OL, zA), _pin(x, R, zA)])
    add([{"type": "axle", "axis": "y", "x": XF, "y": L, "z": zA, "length": 4, "stop": True, "color": gear},
         {"type": "wheel", "axis": "y", "x": XF, "y": Y, "z": zA, "size": WHEEL, "color": "black"}])
    # Handlebar: a 1x9 liftarm across the bike, pinned to a 1x6 Technic brick (along y) on the crown.
    add([{"type": "technic_brick", "w": 1, "l": 6, "x": XF, "y": OL, "z": 16, "color": fork, "hole": "round"},
         {"type": "liftarm", "axis": "y", "x": XF - 1, "y": Y - 3, "z": 16, "length": 9, "color": "dark_gray"},
         _pin(XF - 1, Y - 1, 16, axis="x"), _pin(XF - 1, Y + 3, 16, axis="x")])
    return parts


# ---------------------------------------------------------------------------------------------------- checks

def _body_of(bodies: dict, k: int) -> int:
    return next(r for r, members in bodies.items() if k in members)


def _joint_moving(parts, bodies, joints, part: int):
    """(joint, the side of it that holds `part`, as part indices) for each joint that separates `part`."""
    out = []
    root = _body_of(bodies, part)
    for k, j in enumerate(joints):
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        side = next(s for s in sides if root in s)
        moving = [i for r in side for i in bodies[r]]
        if len(moving) < len(parts) / 2:   # the part's own small side, not the whole bike
            out.append((j, moving))
    return out


def _structure(parts: list[dict]) -> tuple[bool, str]:
    bad = [f"part {i}: {e}" for i, p in enumerate(parts) if (e := _validate(p))]
    if bad:
        return False, "; ".join(bad)
    studs = [p | {"type": PIVOT_AS[p["type"]]} if p["type"] in PIVOT_AS else p for p in structural(parts)]
    studs = [{k: v for k, v in p.items() if k in ("type", "w", "l", "x", "y", "z", "color")} for p in studs]
    issues = [i for i in verify(studs).issues if i.code not in ("floating", "disconnected")]
    if issues:
        return False, "; ".join(i.message for i in issues)
    return True, f"{len(parts)} valid parts; {len(studs)} stud parts break no S3a rule (no overlaps)"


def _connected(parts: list[dict]) -> tuple[bool, str]:
    loose = unseated(parts)
    if loose:
        return False, "; ".join(f"part {i} {why}" for i, why in loose)
    if not connected(parts):
        return False, "not one structure through rigid links and joints"
    bodies, joints = analyse(parts)
    return True, f"one structure: {len(bodies)} bodies, {len(joints)} joints, every pin in two holes"


def _running_gear(parts: list[dict]) -> tuple[bool, str]:
    # A bicycle has one wheel per axle and its kickstand touches the ground: those two car rules do not apply.
    skip = {"too_few_wheels", "body_drags"}
    issues = [i for i in verify_function(parts).issues if i.code not in skip]
    if issues:
        return False, "; ".join(i.message for i in issues)
    return True, "axles turn in two aligned round holes, wheels are held, on their axles and clear everything"


def _in_line(parts: list[dict]) -> tuple[bool, str]:
    wheels = [p for p in parts if p["type"] == "wheel"]
    same = len({(p["axis"], p["y"] if p["axis"] == "y" else p["x"], p["z"]) for p in wheels}) == 1
    return (len(wheels) == 2 and same,
            "2 road wheels, axes parallel (y), in one plane" if same else "the road wheels are not in line")


def _steering(parts: list[dict], degrees: float) -> tuple[bool, str]:
    ok, why = requirement_met(parts, {"kind": "swing", "axis": "vertical", "degrees": degrees, "count": 1})
    if not ok:
        return False, why
    # requirement_met sweeps solid parts only; the steered front wheel must clear the frame too.
    bodies, joints = analyse(parts)
    front = max((i for i, p in enumerate(parts) if p["type"] == "wheel"), key=lambda i: parts[i]["x"])
    for (a, b, kind, point, axis, partners), moving in _joint_moving(parts, bodies, joints, front):
        if abs(abs(axis[2]) - 1) > 1e-6:
            continue
        hits = [sweep(parts, moving, point, axis, s * degrees, ignore=partners, wheels=True) for s in (1, -1)]
        if any(h is not None for h in hits):
            return False, f"the front wheel hits the frame at {min(abs(h) for h in hits if h):.0f} degrees"
        return True, f"fork, handlebar and front wheel turn {degrees} degrees each way about the turntable, clear"
    return False, "no vertical joint carries the front wheel"


def max_steer(parts: list[dict]) -> int:
    """The largest whole-5-degree steering lock that still clears, both ways."""
    best = 0
    for d in range(5, 95, 5):
        if not _steering(parts, d)[0]:
            break
        best = d
    return best


def _crank(parts: list[dict]) -> tuple[bool, str]:
    """The crank axle turns in the frame with both crank arms locked on it, a full turn without hitting anything."""
    bodies, joints = analyse(parts)
    arms = [i for i, p in enumerate(parts) if p["type"] == "liftarm" and "axle_hole" in p]
    if len(arms) != 2:
        return False, f"{len(arms)} crank arm(s); needs 2"
    if _body_of(bodies, arms[0]) != _body_of(bodies, arms[1]):
        return False, "the two crank arms do not turn together (not both locked on the crank axle)"
    for (a, b, kind, point, axis, partners), moving in _joint_moving(parts, bodies, joints, arms[0]):
        if abs(abs(axis[1]) - 1) > 1e-6 or any(parts[i]["type"] == "wheel" for i in moving):
            continue
        hit = sweep(parts, moving, point, axis, 360, steps=72, ignore=partners)
        if hit is not None:
            return False, f"the crank hits the frame at {hit:.0f} degrees"
        return True, "a revolute joint (axle in the bottom bracket's round holes) turns both crank arms 360 degrees clear"
    return False, "the crank arms are not on a joint that turns"


def _mass_points(parts: list[dict]):
    """(centre mm, weight) per part: bricks and beams by their volume, wheels as half-solid cylinders, the small
    connectors by their length."""
    out = []
    for p in parts:
        t = p["type"]
        if t == "wheel":
            r, w = WHEELS[p.get("size", DEFAULT_WHEEL)][0] / 2, wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD
            lo, hi = box(p)
            out.append(((lo + hi) / 2, 0.5 * np.pi * r * r * w))
        elif t in ("axle", "pin", "axle_pin", "axle_joiner", "half_bush", "bush"):
            n = p.get("length", 1)
            c = np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT])
            ax = np.array([t_ == p["axis"] for t_ in "xyz"], dtype=float)
            out.append((c + ax * n * STUD / 2, np.pi * 2.4 ** 2 * n * STUD))
        else:
            lo, hi = box(p)
            out.append(((lo + hi) / 2, float(np.prod(hi - lo)) * (0.6 if t == "liftarm" else 0.5)))
    return out


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) < 3:
        return pts

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and _cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def ground_contacts(parts: list[dict], tol: float = 0.5) -> np.ndarray:
    """Where the build touches the ground (x, y in mm): a tyre on its centre line (a tyre is round across, so
    it is a point, not a patch), other parts by their footprint corners."""
    pts = []
    for p in parts:
        if p["type"] == "wheel":
            lo, hi = box(p)
            if lo[2] <= tol:
                pts.append(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
        elif p["type"] in PIVOT_AS or p["type"] in ("plate", "brick", "technic_brick", "liftarm", "cross_block"):
            lo, hi = box(p)
            if abs(lo[2]) <= tol:
                pts += [(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    return np.array(pts, dtype=float).reshape(-1, 2)


def balance(parts: list[dict], margin: float = BALANCE_MARGIN) -> tuple[bool, str, float]:
    """Static balance of the whole (rigid, joints held) build: its centre of mass must project inside the convex
    hull of its ground contacts, at least `margin` mm from every edge. (The stud force check, stability(), sees only
    studded parts; a pinned Technic frame is one rigid body, so tipping is the question.) Returns (ok, why, margin)."""
    mp = _mass_points(parts)
    com = sum(c * w for c, w in mp) / sum(w for _, w in mp)
    pts = ground_contacts(parts)
    hull = _hull(pts) if len(pts) else pts
    if len(hull) < 3:
        return False, f"{len(pts)} ground contact(s), all in one line: it falls over", -np.inf
    p = com[:2]
    d = min(_cross(b - a, p - a) / float(np.linalg.norm(b - a)) for a, b in zip(hull, np.roll(hull, -1, axis=0)))
    ok = d >= margin
    return ok, (f"centre of mass {d:.1f} mm inside the ground contacts (needs {margin}); "
                f"{len(pts)} contact points"), d


# ---------------------------------------------------------------------------------------------------- entry

def build_bicycle(target: np.ndarray | None = None, length: int = 16) -> dict:
    """The bicycle's parts, the research requirements, and every check as (passed, why)."""
    if target is not None:
        length = _length_from(target)
    parts = parts_for(length)
    no_stand = [p for p in parts if p.get("role") != "kickstand"]
    ok_b, why_b, _ = balance(parts)
    ok_n, why_n, _ = balance(no_stand)
    spin = requirement_met(parts, REQUIREMENTS[0])
    checks = {
        "structure": _structure(parts),
        "connected": _connected(parts),
        "running_gear": _running_gear(parts),
        "spin": spin,
        "wheels_in_line": _in_line(parts),
        "steer": _steering(parts, REQUIREMENTS[1]["degrees"]),
        "crank": _crank(parts),
        "stands": (ok_b, f"double kickstand: {why_b}" + ("" if ok_n else f"; without it: {why_n}")),
    }
    return {"parts": parts, "requirements": REQUIREMENTS, "checks": checks, "unsupported": UNSUPPORTED,
            "length": length, "balance_aid": "double kickstand (a leg each side of the rear stays)"}


def main(out: str = "runs/demo/bicycle/bicycle.json") -> None:
    res = build_bicycle()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res["parts"], indent=1))
    for name, (ok, why) in res["checks"].items():
        print(f"{'PASS' if ok else 'FAIL'} {name}: {why}")
    print("max steering lock:", max_steer(res["parts"]), "degrees")
    print("unsupported:", ", ".join(res["unsupported"]))


if __name__ == "__main__":
    main()
