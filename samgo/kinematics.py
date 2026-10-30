"""Generic motion check (experiment): parts declare connectors; the build's motion follows from how they meet.

No mechanism is special-cased. Each part lists connectors (position in mm, axis, type). Connectors that meet
form joints by type: an axle in a round hole turns (revolute), in an axle hole it locks (rigid); a wheel is
locked on its axle; a Technic pin with friction holds what it pins together (rigid), one without friction lets it
turn (revolute); two pivots on different lines between the same bodies hold them rigid; the two halves of a swivel hinge turn about their shared pin (revolute); studs gripped
from above are rigid. Parts joined rigidly form bodies; joints between bodies are the build's motions. A
requirement ("some body swings 90 degrees about a vertical axis without hitting anything") is checked by
sweeping that body about its joint axis and testing for collisions at each step.

Units: mm, z up. Parts use the stud grid (x, y in studs, z in plates) like the rest of SAMgo.
"""

from dataclasses import dataclass, field

import numpy as np

from .technic import HOLE_HEIGHT, LENGTH, PLATE, STUD, WHEELS, DEFAULT_WHEEL, extent, wheel_studs

HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3, "hinge_base": 1, "hinge_top": 1, "tile": 1, "slope": 3,
          "invslope": 3, "curve": 2, "roundbrick": 3, "roundplate": 1, "arch": 3,
          "car_door": 3, "click_a": 1, "click_b": 1, "turntable_base": 1, "turntable_top": 1}
# Real parts behind the pivoting kinds (LDraw): swivel hinge 2429/2430, car door 3821/3822 (1x3x1),
# click hinge plates 44301/44302, turntable 2x2 3680/3679.
TOL = 0.5  # mm
BEAMS = {"liftarm", "cross_block"}  # Technic parts with a body that can collide but no studs to grip by
AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0.0, 1, 0]), "z": np.array([0.0, 0, 1])}


@dataclass
class Connector:
    kind: str           # round_hole | axle_hole | axle | hub | pin | pin_free | hinge_pin_a | hinge_pin_b ...
    point: np.ndarray   # a point on the axis, mm
    axis: np.ndarray    # unit vector
    span: tuple = (0.0, 0.0)  # extent along the axis from `point` (axles, holes)


@dataclass
class Body:
    parts: set = field(default_factory=set)


def box(p: dict) -> tuple[np.ndarray, np.ndarray]:
    if p["type"] in BEAMS:
        lo, hi = extent(p)
        return np.array(lo), np.array(hi)
    lo = np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE])
    return lo, lo + np.array([p["w"] * STUD, p["l"] * STUD, HEIGHT[p["type"]] * PLATE])


def connectors(p: dict) -> list[Connector]:
    """Connectors of a part, from its LDraw geometry (pivots measured from the official parts library)."""
    t = p["type"]
    y_axis, z_axis = np.array([0.0, 1, 0]), np.array([0.0, 0, 1])
    if t == "technic_brick":  # 1xN: N-1 holes across it, 5.6 mm up, at the stud edges
        kind = "axle_hole" if p.get("hole") == "axle" else "round_hole"
        h = p["z"] * PLATE + HOLE_HEIGHT
        if p["w"] > 1:  # along x: holes along y
            return [Connector(kind, np.array([(p["x"] + i) * STUD, p["y"] * STUD, h]), y_axis, (0.0, STUD))
                    for i in range(1, p["w"])]
        return [Connector(kind, np.array([p["x"] * STUD, (p["y"] + i) * STUD, h]), AXES["x"], (0.0, STUD))
                for i in range(1, p["l"])]
    if t == "axle":
        return [Connector("axle", _start(p), AXES[p.get("axis", "y")], (0.0, p["length"] * STUD))]
    if t == "wheel":
        return [Connector("hub", _start(p), AXES[p.get("axis", "y")],
                          (0.0, wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD))]
    if t == "pin":  # 2780 / 6558 grip the hole (friction); 3673 turns in it
        kind = "pin" if p.get("friction", True) else "pin_free"
        return [Connector(kind, _start(p), AXES[p["axis"]], (0.0, p.get("length", 2) * STUD))]
    if t == "axle_pin":  # 43093: pin half (with friction) and axle half, a stud each
        pin, axle = ((0.0, STUD), (STUD, 2 * STUD)) if p.get("axle_end", "end") == "end" else ((STUD, 2 * STUD), (0.0, STUD))
        return [Connector("pin", _start(p), AXES[p["axis"]], pin), Connector("axle", _start(p), AXES[p["axis"]], axle)]
    if t in ("axle_joiner", "half_bush"):  # 6538a: two axles in line in one axle-hole tube; 32123a grips its axle
        return [Connector("axle_hole", _start(p), AXES[p["axis"]], (0.0, LENGTH[t] * STUD))]
    if t == "liftarm":  # thick beam: a round hole across it on every stud edge along it
        along, across = AXES[p["axis"]], AXES["y" if p["axis"] == "x" else "x"]
        return [Connector("round_hole", _start(p) + i * STUD * along, across, (0.0, STUD)) for i in range(p["length"])]
    if t == "cross_block":  # 6536: the axle hole on the axis line, a pin hole 1 stud toward pin_at, across both
        ax = AXES[p["axis"]]
        side = p.get("pin_at", "+z")
        off = AXES[side[1]] * (STUD if side[0] == "+" else -STUD)
        pin_axis = np.abs(np.cross(ax, AXES[side[1]]))
        centre = _start(p) + ax * STUD / 2 + off
        return [Connector("axle_hole", _start(p), ax, (0.0, STUD)),
                Connector("round_hole", centre - pin_axis * STUD / 2, pin_axis, (0.0, STUD))]
    if t in ("hinge_base", "hinge_top"):  # swivel hinge 2429/2430: pin vertical, on the long edge at the joint
        pin = np.array([p["pin_x"] * STUD, p["pin_y"] * STUD, p["z"] * PLATE])
        return [Connector("hinge_pin_a" if t == "hinge_base" else "hinge_pin_b", pin, z_axis, (0.0, PLATE))]
    if t == "car_door":  # 3821/3822: turns about a vertical axis through the stud at its hinge end
        pin = np.array([(p["pin_x"] + 0.5) * STUD, (p["pin_y"] + 0.5) * STUD, p["z"] * PLATE])
        return [Connector("door_pin", pin, z_axis, (-PLATE, 3 * PLATE))]
    if t in ("click_a", "click_b"):  # click hinge plates 44301/44302: a horizontal pin along `pin_axis`
        ax = {"x": np.array([1.0, 0, 0]), "y": y_axis}[p.get("pin_axis", "x")]
        pin = np.array([p["pin_x"] * STUD, p["pin_y"] * STUD, p["z"] * PLATE + 2 / 20 * STUD])
        return [Connector("click_pin_a" if t == "click_a" else "click_pin_b", pin, ax, (0.0, STUD))]
    if t in ("turntable_base", "turntable_top"):  # turntable 3680/3679: turns about the vertical axis at its center
        c = np.array([(p["x"] + p["w"] / 2) * STUD, (p["y"] + p["l"] / 2) * STUD, p["z"] * PLATE])
        return [Connector("turn_a" if t == "turntable_base" else "turn_b", c, z_axis, (0.0, 2 * PLATE))]
    return []


def _start(p: dict) -> np.ndarray:
    """Start of an axis-line part (axle, wheel, pin...): on the stud edge at Technic hole height; an upright (z)
    pin stands at stud point (x, y) from the bottom of plate z."""
    if p.get("axis") == "z":
        return np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE])
    return np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT])


def _on_axis(a: Connector, b: Connector) -> bool:
    """Same axis line, overlapping spans."""
    if abs(abs(float(a.axis @ b.axis)) - 1) > 1e-6:
        return False
    d = b.point - a.point
    off = d - (d @ a.axis) * a.axis
    if np.linalg.norm(off) > TOL:
        return False
    s0, s1 = a.span
    t0, t1 = sorted((float(d @ a.axis) + b.span[0], float(d @ a.axis) + b.span[1]))
    # A pin must reach into the hole: a pin end only touching a hole's face holds nothing.
    if {a.kind, b.kind} & {"pin", "pin_free"}:
        return min(s1, t1) - max(s0, t0) > TOL
    return min(s1, t1) - max(s0, t0) > -TOL


def _same_line(a: tuple, b: tuple) -> bool:
    """Do two (point, axis) pivots turn about one line?"""
    (pa, ua), (pb, ub) = a, b
    if abs(abs(float(ua @ ub)) - 1) > 1e-6:
        return False
    d = pb - pa
    return bool(np.linalg.norm(d - (d @ ua) * ua) <= TOL)


JOINT = {  # (kind, kind) -> joint type
    frozenset(["axle", "round_hole"]): "revolute",
    frozenset(["axle", "axle_hole"]): "rigid",
    frozenset(["axle", "hub"]): "rigid",
    frozenset(["pin", "round_hole"]): "rigid",
    frozenset(["pin_free", "round_hole"]): "revolute",
    frozenset(["hinge_pin_a", "hinge_pin_b"]): "revolute",
    frozenset(["click_pin_a", "click_pin_b"]): "revolute",
    frozenset(["turn_a", "turn_b"]): "revolute",
    frozenset(["door_pin", "stud"]): "revolute",
}
PIVOTS = {"car_door"}  # parts that turn on a stud instead of gripping it


def _studs_rigid(a: dict, b: dict) -> bool:
    """One part sits directly on the other and they share a stud cell (stud grip: rigid)."""
    (alo, ahi), (blo, bhi) = box(a), box(b)
    touching = abs(ahi[2] - blo[2]) < TOL or abs(bhi[2] - alo[2]) < TOL
    overlap = min(ahi[0], bhi[0]) - max(alo[0], blo[0]) > TOL and min(ahi[1], bhi[1]) - max(alo[1], blo[1]) > TOL
    return touching and overlap


def analyse(parts: list[dict]):
    """(bodies as lists of part indices, joints as (body a, body b, type, point, axis))."""
    n = len(parts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cons = [connectors(p) for p in parts]
    links = []
    turning = set()  # pairs joined by a rotating connector: their touching faces turn, they are not stud-locked
    for i in range(n):
        for j in range(i + 1, n):
            for a in cons[i]:
                for b in cons[j]:
                    if JOINT.get(frozenset([a.kind, b.kind])) == "revolute" and _on_axis(a, b):
                        turning.add((i, j))
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = parts[i], parts[j]
            if (i, j) not in turning and pi["type"] in HEIGHT and pj["type"] in HEIGHT and _studs_rigid(pi, pj):
                pivot = pi if pi["type"] in PIVOTS else pj if pj["type"] in PIVOTS else None
                if pivot is None:
                    links.append((i, j, "rigid", None, None))
                else:  # a door sits on the stud it turns about: that stud is its axle
                    c = connectors(pivot)[0]
                    other = pj if pivot is pi else pi
                    lo, hi = box(other)
                    if lo[0] - TOL <= c.point[0] <= hi[0] + TOL and lo[1] - TOL <= c.point[1] <= hi[1] + TOL:
                        links.append((i, j, "revolute", c.point, c.axis))
            for a in cons[i]:
                for b in cons[j]:
                    kind = JOINT.get(frozenset([a.kind, b.kind]))
                    if kind and _on_axis(a, b):
                        links.append((i, j, kind, a.point, a.axis))
    # A pin without friction turns in every hole it is in; it rides with its first holder, so the joint lies
    # between the two parts it pins together.
    for k, p in enumerate(parts):
        if p.get("type") == "pin" and not p.get("friction", True):
            mine = [m for m, (i, j, kind, _, _) in enumerate(links) if k in (i, j) and kind == "revolute"]
            if mine:
                i, j, _, point, axis = links[mine[0]]
                links[mine[0]] = (i, j, "rigid", point, axis)
    for i, j, kind, _, _ in links:
        if kind == "rigid":
            parent[find(i)] = find(j)
    # Two bodies turning on two different lines (two free pins in two holes, an axle and a pin) cannot turn.
    merged = True
    while merged:
        merged = False
        lines: dict[tuple, list] = {}
        for i, j, kind, point, axis in links:
            if kind == "revolute" and find(i) != find(j):
                lines.setdefault(tuple(sorted((find(i), find(j)))), []).append((point, axis))
        for (ra, rb), ls in lines.items():
            if any(not _same_line(ls[0], l) for l in ls[1:]):
                parent[find(ra)] = find(rb)
                merged = True
                break
    bodies: dict[int, list[int]] = {}
    for i in range(n):
        bodies.setdefault(find(i), []).append(i)
    joints = {}
    for i, j, kind, point, axis in links:
        if kind == "revolute" and find(i) != find(j):
            key = tuple(sorted((find(i), find(j))))
            # The two parts meeting at the joint (hinge halves, axle and hole) turn past each other by design.
            prev = joints.get(key)
            partners = (prev[5] if prev else set()) | {i, j}
            joints[key] = (key[0], key[1], "revolute", point, axis, partners)
    return bodies, list(joints.values())


def _corners(p: dict) -> np.ndarray:
    lo, hi = box(p)
    if p["type"] == "wheel":
        r = WHEELS[p.get("size", DEFAULT_WHEEL)][0] / 2
        c = np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT])
        w = wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD
        lo, hi = c - [r, 0, r], c + [r, w, r]
    xs, ys, zs = np.meshgrid(*[np.linspace(lo[k], hi[k], 4) for k in range(3)], indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), zs.ravel()]


def _inside(pts: np.ndarray, p: dict) -> bool:
    lo, hi = box(p)
    return bool(((pts > lo + TOL) & (pts < hi - TOL)).all(axis=1).any())


def sweep(parts: list[dict], moving: list[int], point: np.ndarray, axis: np.ndarray, degrees: float,
          steps: int = 18, ignore: set = frozenset()) -> float | None:
    """Turn the parts `moving` about the axis through `point`; the first angle where they hit a fixed part
    (None if the whole sweep is clear). `ignore`: the joint's own parts, which turn past each other."""
    solid = HEIGHT.keys() | BEAMS
    fixed = [p for k, p in enumerate(parts) if k not in moving and k not in ignore and p["type"] in solid]
    pts = np.vstack([_corners(parts[k]) for k in moving if parts[k]["type"] in solid])
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    for s in range(1, steps + 1):
        th = np.radians(degrees * s / steps)
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        moved = (pts - point) @ R.T + point
        if any(_inside(moved, f) for f in fixed):
            return degrees * s / steps
    return None


# How many parts each Technic connector must engage to hold anything: a pin joins two holes; a joiner or half
# bush sits on an axle.
SEATS = {"pin": 2, "axle_pin": 2, "axle_joiner": 1, "half_bush": 1}


def unseated(parts: list[dict]) -> list[tuple[int, str]]:
    """(part index, why) for each Technic connector not engaged with enough parts (a pin not in two holes)."""
    cons = [connectors(p) if isinstance(p, dict) and p.get("type") else [] for p in parts]
    out = []
    for k, p in enumerate(parts):
        need = SEATS.get(p.get("type")) if isinstance(p, dict) else None
        if need is None:
            continue
        held = {j for j in range(len(parts)) if j != k for a in cons[k] for b in cons[j]
                if JOINT.get(frozenset([a.kind, b.kind])) and _on_axis(a, b)}
        if len(held) < need:
            what = "holes" if p["type"] in ("pin", "axle_pin") else "axle"
            out.append((k, f"is engaged with {len(held)} part(s); it needs {need} ({what})"))
    return out


GROUND = {"ground"}


def requirement_met(parts: list[dict], req: dict) -> tuple[bool, str]:
    """Check one requirement against the build's motions, with no mechanism-specific code.

    req kinds:
      {"kind": "spin", "count": 4, "touch_ground": true}: at least `count` wheels on revolute joints, all axes
          parallel, wheels the lowest parts;
      {"kind": "swing", "axis": "vertical"|"horizontal", "degrees": 90}: some body turns on a joint with that
          axis, through `degrees` in one direction, without hitting anything.
    """
    bodies, joints = analyse(parts)
    if req["kind"] == "spin":
        wheels = [i for i, p in enumerate(parts) if p.get("type") == "wheel"]
        if len(wheels) < req.get("count", 2):
            return False, f"{len(wheels)} wheel(s); needs {req.get('count', 2)}"
        root_of = {i: r for r, members in bodies.items() for i in members}
        spinning = [w for w in wheels if any(root_of[w] in (j[0], j[1]) for j in joints)]
        if len(spinning) < req.get("count", 2):
            return False, f"{len(spinning)} of {len(wheels)} wheels can spin (the rest are locked or unsupported)"
        axes = {tuple(np.round(np.abs(j[4]), 3)) for j in joints if root_of[spinning[0]] in (j[0], j[1])}
        if len({tuple(np.round(np.abs(j[4]), 3)) for j in joints}) > 1 and req.get("parallel", True):
            return False, "wheel axes are not parallel"
        return True, "wheels spin"
    want = np.array([0.0, 0, 1]) if req["axis"] == "vertical" else None
    best = "no joint with that axis"
    for a, b, kind, point, axis, partners in joints:
        vertical = abs(abs(axis[2]) - 1) < 1e-6
        if (req["axis"] == "vertical") != vertical:
            continue
        # The moving side is the body that does not hold the lowest part (the base).
        lowest = min(range(len(parts)), key=lambda i: parts[i].get("z", 0))
        moving = bodies[b] if lowest in bodies[a] else bodies[a]
        for sign in (1, -1):
            hit = sweep(parts, moving, point, axis, sign * req["degrees"], ignore=partners)
            if hit is None:
                return True, f"turns {req['degrees']} degrees clear"
            best = f"hits something at {abs(hit):.0f} degrees"
    return False, best


def connected(parts: list[dict]) -> bool:
    """One structure through rigid links and joints together, standing on the ground (z = 0) or on wheels: the
    structural check for builds with moving parts, where a hinged lid hangs on its hinge, not on studs."""
    n = len(parts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bodies, joints = analyse(parts)
    for members in bodies.values():
        for k in members[1:]:
            parent[find(k)] = find(members[0])
    for a, b, *_ in joints:
        parent[find(a)] = find(b)
    roots = {find(i) for i in range(n)}
    grounded = any(parts[i].get("z", 0) == 0 or parts[i].get("type") == "wheel" for i in range(n))
    return len(roots) == 1 and grounded


def locked(parts: list[dict], as_type: dict) -> list[dict]:
    """The build with every joint held in its current pose, for the static force check: the two halves of a
    hinge lying side by side at one height become one plate (click hinges hold their angle; swivel hinges by
    friction); other pivoting parts become the plain parts they occupy."""
    out, used = [], set()
    for i, p in enumerate(parts):
        if i in used or p.get("type") not in as_type:
            continue
        for j in range(i + 1, len(parts)):
            q = parts[j]
            if j in used or q.get("type") not in as_type or q["z"] != p["z"]:
                continue
            xs = sorted((p["x"], p["x"] + p["w"], q["x"], q["x"] + q["w"]))
            ys = sorted((p["y"], p["y"] + p["l"], q["y"], q["y"] + q["l"]))
            side_by_side = (p["x"] == q["x"] and p["w"] == q["w"] and (p["y"] + p["l"] == q["y"] or q["y"] + q["l"] == p["y"])) \
                or (p["y"] == q["y"] and p["l"] == q["l"] and (p["x"] + p["w"] == q["x"] or q["x"] + q["w"] == p["x"]))
            if side_by_side:
                used |= {i, j}
                out.append({"type": as_type[p["type"]], "w": xs[-1] - xs[0], "l": ys[-1] - ys[0], "x": xs[0],
                            "y": ys[0], "z": p["z"], "color": p.get("color", "dark_gray")})
                break
    return out + [p if p.get("type") not in as_type else p | {"type": as_type[p["type"]]}
                  for k, p in enumerate(parts) if k not in used]
