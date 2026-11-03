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
          "car_door": 3, "click_a": 1, "click_b": 1, "turntable_base": 1, "turntable_top": 1,
          "bar_plate": 1, "clip_plate": 1}
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
    if p["type"] == "wheel":  # the tyre's bounding box
        r = WHEELS[p.get("size", DEFAULT_WHEEL)][0] / 2
        w = wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD
        c = _start(p)
        along = AXES[p.get("axis", "y")]
        lo = c - r * (1 - along)
        return lo, lo + 2 * r * (1 - along) + w * along
    if p["type"] in BEAMS:
        lo, hi = extent(p)
        return np.array(lo), np.array(hi)
    lo = np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE])
    return lo, lo + np.array([p["w"] * STUD, p["l"] * STUD, HEIGHT[p["type"]] * PLATE])


def clip_axis(p: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(axis, start point, length) of the hinge line of a bar plate (60478) or clip plate (63868), measured from
    the LDraw parts: the bar / clip centre is 1.5 studs from the plate centre, i.e. half a stud past the end it
    faces, 2 LDU (0.8 mm) below the plate's top, across the plate's width. A bar plate and a clip plate at one
    height, facing each other with one stud between their ends, share this line."""
    f = p.get("facing", "+x")
    top = (p["z"] + 1) * PLATE - 0.8
    if f in ("+x", "-x"):
        x = (p["x"] + p["w"] + 0.5) if f == "+x" else (p["x"] - 0.5)
        return np.array([0.0, 1, 0]), np.array([x * STUD, p["y"] * STUD, top]), p["l"] * STUD
    y = (p["y"] + p["l"] + 0.5) if f == "+y" else (p["y"] - 0.5)
    return np.array([1.0, 0, 0]), np.array([p["x"] * STUD, y * STUD, top]), p["w"] * STUD


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
    if t == "bush":  # 3713: a 1-stud axle-hole tube that grips its axle (the cars' wheel retainers)
        return [Connector("axle_hole", _start(p), AXES[p["axis"]], (0.0, STUD))]
    if t == "liftarm":  # thick beam: a round hole across it on every stud edge along it
        along, across = AXES[p["axis"]], AXES["y" if p["axis"] == "x" else "x"]
        n = p["length"]
        axle_at = {"start": 0, "end": n - 1}.get(p.get("axle_hole"))  # 60483: one end hole is an axle hole
        return [Connector("axle_hole" if i == axle_at else "round_hole", _start(p) + i * STUD * along, across,
                          (0.0, STUD)) for i in range(n)]
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
    if t in ("bar_plate", "clip_plate"):  # 60478 / 63868: bar or clip past the end the plate faces
        ax, end, line = clip_axis(p)
        return [Connector("bar_end" if t == "bar_plate" else "clip_end", end, ax, (0.0, line))]
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
    frozenset(["bar_end", "clip_end"]): "revolute",
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


def _near_pairs(parts: list[dict], margin: float = 2 * STUD) -> list[tuple[int, int]]:
    """Pairs (i < j) whose bounding boxes, grown by `margin` (connectors reach at most about a stud past a part's
    box: a bar or clip end, a pin, a stud), overlap. Same answers as testing all pairs, without the n^2 Python loop."""
    n = len(parts)
    if n < 2:
        return []
    boxes = [box(p) if p.get("type") in HEIGHT or p.get("type") == "wheel" or p.get("type") in BEAMS else None
             for p in parts]
    lo = np.array([b[0] if b is not None else (-1e9, -1e9, -1e9) for b in boxes], dtype=float) - margin
    hi = np.array([b[1] if b is not None else (1e9, 1e9, 1e9) for b in boxes], dtype=float) + margin
    out = []
    for i in range(n - 1):
        ok = np.all((lo[i + 1:] <= hi[i]) & (hi[i + 1:] >= lo[i]), axis=1)
        out.extend((i, i + 1 + int(k)) for k in np.flatnonzero(ok))
    return out


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
    near = _near_pairs(parts)  # every pair that can touch or share a connector; the rest are far apart
    turning = set()  # pairs joined by a rotating connector: their touching faces turn, they are not stud-locked
    for i, j in near:
        for a in cons[i]:
            for b in cons[j]:
                if JOINT.get(frozenset([a.kind, b.kind])) == "revolute" and _on_axis(a, b):
                    turning.add((i, j))
    for i, j in near:
        if True:
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
            if any(not _same_line(ls[0], line) for line in ls[1:]):
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
    if p["type"] == "wheel":  # points on and inside the tyre's outline, across its width
        r = WHEELS[p.get("size", DEFAULT_WHEEL)][0] / 2
        along = AXES[p.get("axis", "y")]
        u, v = (AXES["x"], AXES["z"]) if p.get("axis", "y") == "y" else (AXES["y"], AXES["z"])
        c, w = _start(p), wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = [c + k * r * (np.cos(t) * u + np.sin(t) * v) for k in (1.0, 0.66, 0.33) for t in th] + [c]
        return np.array([q + s * w * along for q in ring for s in np.linspace(0, 1, 4)])
    # Points about every 4 mm over the box, plus the midpoints between them: corner-only points of a 1-plate part
    # all lie on the faces of a part on the same layer, where _inside's tolerance never counts them, so a plate
    # swinging into a plate of its own layer would never be seen.
    def axis_points(k):
        v = np.linspace(lo[k], hi[k], max(2, int(np.ceil((hi[k] - lo[k]) / 4)) + 1))
        return np.concatenate([v, (v[:-1] + v[1:]) / 2])
    xs, ys, zs = np.meshgrid(*[axis_points(k) for k in range(3)], indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), zs.ravel()]


def _inside(pts: np.ndarray, p: dict) -> bool:
    lo, hi = box(p)
    return bool(((pts > lo + TOL) & (pts < hi - TOL)).all(axis=1).any())


def sweep(parts: list[dict], moving: list[int], point: np.ndarray, axis: np.ndarray, degrees: float,
          steps: int = 18, ignore: set = frozenset(), wheels: bool = False) -> float | None:
    """Turn the parts `moving` about the axis through `point`; the first angle where they hit a fixed part
    (None if the whole sweep is clear). `ignore`: the joint's own parts, which turn past each other. With
    `wheels`, tyres count too (as their bounding boxes): a steered wheel must clear the frame."""
    solid = HEIGHT.keys() | BEAMS | ({"wheel"} if wheels else set())
    fixed = [p for k, p in enumerate(parts) if k not in moving and k not in ignore and p["type"] in solid]
    corners = [_corners(parts[k]) for k in moving if parts[k]["type"] in solid]
    if not corners:  # nothing solid turns (a bare axle with its wheels): nothing can hit
        return None
    pts = np.vstack(corners)
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    steps = max(steps, int(abs(degrees) / 5))  # at most 5 degrees a step
    for s in range(steps + 1):
        th = np.radians(degrees * s / steps)
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        moved = (pts - point) @ R.T + point
        if any(_inside(moved, f) for f in fixed):
            return degrees * s / steps
    return None


# How many parts each Technic connector must engage to hold anything: a pin joins two holes; a joiner or half
# bush sits on an axle.
SEATS = {"pin": 2, "axle_pin": 2, "axle_joiner": 1, "half_bush": 1, "bush": 1}


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


def _sides(bodies: dict, joints: list, k: int) -> tuple[set, set] | None:
    """The two groups of bodies joint k separates (following every other joint); None if they stay joined."""
    a, b = joints[k][0], joints[k][1]
    links: dict = {}
    for n, j in enumerate(joints):
        if n != k:
            links.setdefault(j[0], set()).add(j[1])
            links.setdefault(j[1], set()).add(j[0])
    def reach(start):
        seen, todo = {start}, [start]
        while todo:
            for nxt in links.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    side_b = reach(b)
    if a in side_b:
        return None
    return reach(a), side_b  # bodies joined to neither side (loose parts) stay where they are


def _outward(parts: list[dict], moving: list[int], point: np.ndarray, axis: np.ndarray, sign: int) -> bool:
    """Whether turning `moving` a little in direction `sign` moves its centre away from the model's centre."""
    def pts(ids):
        return np.vstack([_corners(parts[i]) for i in ids if parts[i].get("type") in HEIGHT.keys() | BEAMS])
    others = [i for i in range(len(parts)) if i not in moving]
    mine = pts(moving)
    middle = pts(others).mean(axis=0) if others else point
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.radians(10 * sign)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    before, after = mine.mean(axis=0), ((mine - point) @ R.T + point).mean(axis=0)
    return np.linalg.norm(after - middle) > np.linalg.norm(before - middle)


def wheel_axis_matches(part, point, axis):
    """A joint must turn a wheel about its own axle line, not merely carry it around."""
    own = AXES[part.get("axis", "y")]
    center = np.array([part["x"] * STUD, part["y"] * STUD, part["z"] * PLATE + HOLE_HEIGHT])
    return (np.linalg.norm(np.cross(own, axis)) < 1e-6
            and np.linalg.norm(np.cross(center - point, own)) < 1e-6)


def bound_motion(parts, members, degrees, axis):
    """Sample a rotation of exactly the named payload, including either side of a joint."""
    if axis not in ("vertical", "horizontal") or not np.isfinite(degrees) or degrees <= 0:
        raise ValueError("motion needs a vertical/horizontal axis and positive finite degrees")
    bodies, joints = analyse(parts)
    best = "no matching joint moves the specified payload"
    for k, (_, _, _, point, direction, partners) in enumerate(joints):
        if (abs(direction[2]) > 0.9) != (axis == "vertical"):
            continue
        sides = _sides(bodies, joints, k)
        if sides is None:
            continue
        for side in sides:
            moving = {i for body in side for i in bodies[body]}
            if moving != set(members):
                continue
            for sign in (1, -1):
                hit = sweep(parts, sorted(moving), point, direction, sign * degrees, ignore=partners,
                            wheels=True)
                if hit is None:
                    return True, "specified payload clears its sampled rotation"
                best = f"specified payload hits something at {abs(hit):.0f} degrees"
    return False, best


def requirement_met(parts: list[dict], req: dict) -> tuple[bool, str]:
    """Check one requirement against the build's motions, with no mechanism-specific code.

    req kinds:
      {"kind": "spin", "count": 4, "parallel": true}: at least `count` wheels on revolute joints; with
          "parallel" (the default, for cars) the spinning wheels' axes must all be parallel;
      {"kind": "swing", "axis": "vertical"|"horizontal", "degrees": 90, "count": 1}: at least `count` joints
          with that axis (and, with "below": Z, a pivot below plate Z) each let what they carry turn through
          `degrees` in one direction without hitting
          anything. What turns is the smaller side of the joint with everything hanging off it (a caster's fork
          with its wheel, a door, a seat): turning it relative to the rest is the same motion.
    """
    bodies, joints = analyse(parts)
    root_of = {i: r for r, members in bodies.items() for i in members}
    need = req.get("count", 2 if req["kind"] == "spin" else 1)
    if req["kind"] == "spin":
        wheels = [i for i, p in enumerate(parts) if p.get("type") == "wheel"]
        if len(wheels) < need:
            return False, f"{len(wheels)} wheel(s); needs {need}"
        spinning = [w for w in wheels if any(
            root_of[w] in (j[0], j[1]) and wheel_axis_matches(parts[w], j[3], j[4])
            and _sides(bodies, joints, k) is not None for k, j in enumerate(joints))]
        if len(spinning) < need:
            return False, f"{len(spinning)} of {len(wheels)} wheels can spin (the rest are locked or unsupported)"
        axes = {parts[w].get("axis", "y") for w in spinning}
        if len(axes) > 1 and req.get("parallel", True):
            return False, "wheel axes are not parallel"
        return True, f"{len(spinning)} wheels spin"
    best, passed = "no joint with that axis", 0
    for k, (a, b, kind, point, axis, partners) in enumerate(joints):
        vertical = abs(abs(axis[2]) - 1) < 1e-6
        if (req["axis"] == "vertical") != vertical:
            continue
        if "below" in req and point[2] >= req["below"] * PLATE:
            continue  # only joints whose pivot is lower than plate `below` (a chair's casters, not its seat)
        sides = _sides(bodies, joints, k)
        if sides is None:
            best = "the joint is locked by another link"
            continue
        small = min(sides, key=lambda g: sum(len(bodies[r]) for r in g))
        moving = [i for r in small for i in bodies[r]]
        if not any(parts[i]["type"] in HEIGHT.keys() | BEAMS for i in moving):
            continue  # an axle turning with its wheels: that is a spin, not a swing
        signs = (1, -1)
        if req.get("outward"):  # only the way that carries it away from the model's middle (doors open out)
            signs = [sg for sg in signs if _outward(parts, moving, point, axis, sg)]
        for sign in signs:
            hit = sweep(parts, moving, point, axis, sign * req["degrees"], ignore=partners)
            if hit is None:
                passed += 1
                break
            best = f"hits something at {abs(hit):.0f} degrees"
    if passed >= need:
        return True, f"{passed} joint(s) turn {req['degrees']} degrees clear"
    return False, (f"{passed} of {need} joints turn {req['degrees']} degrees clear; " if passed else "") + best


def rotors(parts: list[dict], axis: str = "horizontal") -> tuple[int, str]:
    """Rotors: an axle turning in round holes with something locked on it (an axle hole: a liftarm, a cross block,
    a hub) that turns all the way round clear of every other part and above the model's lowest point (the table).
    Parts spinning loose on the axle, or an axle with nothing on it, are not rotors. Returns (count, why)."""
    bodies, joints = analyse(parts)
    ground = min((box(p)[0][2] for p in parts if p.get("type") in HEIGHT), default=0.0)
    count, why = 0, "no axle turns in a round hole"
    for k, (a, b, kind, point, ax, partners) in enumerate(joints):
        if kind != "revolute" or (abs(ax[2]) > 0.9) != (axis == "vertical"):
            continue
        sides = _sides(bodies, joints, k)
        if sides is None:
            why = "the axle is locked by another link"
            continue
        small = min(sides, key=lambda g: sum(len(bodies[r]) for r in g))
        moving = [i for r in small for i in bodies[r]]
        locked = [bodies[r] for r in small if any(parts[i].get("type") == "axle" for i in bodies[r])]
        if not locked:
            continue  # a part turning loose on an axle, a hinge: not a rotor
        if not {parts[i].get("type") for m in locked for i in m} & (HEIGHT.keys() | BEAMS):
            why = "nothing is locked on the axle"
            continue
        if sweep(parts, moving, point, ax, 360, steps=24, ignore=partners) is not None:
            why = "what is on the axle hits a part when it turns"
            continue
        n = np.asarray(ax, float) / np.linalg.norm(ax)
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        pts = np.vstack([_corners(parts[i]) for i in moving if parts[i].get("type") in HEIGHT.keys() | BEAMS])
        low = min(((pts - point) @ (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K).T + point)[:, 2].min()
                  for t in np.radians(np.arange(0, 360, 15)))
        if low < ground - TOL:
            why = "what is on the axle reaches below the table when it turns: put the axle higher"
            continue
        count += 1
    return count, (f"{count} rotor(s) turn all the way round" if count else why)


JOINT_HALVES = {"bar_plate", "clip_plate", "hinge_base", "hinge_top", "click_a", "click_b", "turntable_base",
                "turntable_top"}


def dangling(parts: list[dict], bodies: dict | None = None) -> list[int]:
    """Joint halves that hold nothing: a hinge, clip or turntable half whose studs grip no other part, so in real
    LEGO it only hangs on its partner (a clip on a bar over empty space). Each half must be built into a side."""
    if bodies is None:
        bodies, _ = analyse(parts)
    return sorted(m[0] for m in bodies.values() if len(m) == 1 and parts[m[0]].get("type") in JOINT_HALVES)


def groups(parts: list[dict]) -> int:
    """How many separate structures (through studs and joints), ground or not."""
    parent = list(range(len(parts)))

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
    return len({find(i) for i in range(len(parts))})


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
    return len(roots) == 1 and grounded and not dangling(parts, bodies)


def locked(parts: list[dict], as_type: dict) -> list[dict]:
    """The build with every joint held in its current pose, for the static force check: the two halves of a
    hinge lying side by side at one height become one plate (click hinges hold their angle; swivel hinges by
    friction); other pivoting parts become the plain parts they occupy."""
    out, used = [], set()
    clip = ("bar_plate", "clip_plate")  # these pair across their one-stud gap, below, never side by side
    for i, p in enumerate(parts):
        if i in used or p.get("type") not in as_type or p["type"] in clip:
            continue
        for j in range(i + 1, len(parts)):
            q = parts[j]
            if j in used or q.get("type") not in as_type or q["type"] in clip or q["z"] != p["z"]:
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
    for i, p in enumerate(parts):  # a clip on its bar, one stud apart: held still, one rigid plate across the gap
        if i in used or p.get("type") != "bar_plate":
            continue
        for j, q in enumerate(parts):
            if j in used or q.get("type") != "clip_plate" or q["z"] != p["z"]:
                continue
            (a1, s1, _), (a2, s2, _) = clip_axis(p), clip_axis(q)
            if abs(abs(float(a1 @ a2)) - 1) < 1e-6 and np.linalg.norm((s1 - s2) - ((s1 - s2) @ a1) * a1) <= TOL:
                xs = sorted((p["x"], p["x"] + p["w"], q["x"], q["x"] + q["w"]))
                ys = sorted((p["y"], p["y"] + p["l"], q["y"], q["y"] + q["l"]))
                used |= {i, j}
                out.append({"type": "plate", "w": xs[-1] - xs[0], "l": ys[-1] - ys[0], "x": xs[0], "y": ys[0],
                            "z": p["z"], "color": p.get("color", "dark_gray")})
                break
    return out + [p if p.get("type") not in as_type else p | {"type": as_type[p["type"]]}
                  for k, p in enumerate(parts) if k not in used]
