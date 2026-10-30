"""Generic motion check (experiment): parts declare connectors; the build's motion follows from how they meet.

No mechanism is special-cased. Each part lists connectors (position in mm, axis, type). Connectors that meet
form joints by type: an axle in a round hole turns (revolute), in an axle hole it locks (rigid); a wheel is
locked on its axle; the two halves of a swivel hinge turn about their shared pin (revolute); studs gripped
from above are rigid. Parts joined rigidly form bodies; joints between bodies are the build's motions. A
requirement ("some body swings 90 degrees about a vertical axis without hitting anything") is checked by
sweeping that body about its joint axis and testing for collisions at each step.

Units: mm, z up. Parts use the stud grid (x, y in studs, z in plates) like the rest of SAMgo.
"""

from dataclasses import dataclass, field

import numpy as np

from .technic import HOLE_HEIGHT, PLATE, STUD, WHEELS, DEFAULT_WHEEL, wheel_studs

HEIGHT = {"plate": 1, "brick": 3, "technic_brick": 3, "hinge_base": 1, "hinge_top": 1}
TOL = 0.5  # mm


@dataclass
class Connector:
    kind: str           # round_hole | axle_hole | axle | hub | hinge_pin_a | hinge_pin_b
    point: np.ndarray   # a point on the axis, mm
    axis: np.ndarray    # unit vector
    span: tuple = (0.0, 0.0)  # extent along the axis from `point` (axles, holes)


@dataclass
class Body:
    parts: set = field(default_factory=set)


def box(p: dict) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE])
    return lo, lo + np.array([p["w"] * STUD, p["l"] * STUD, HEIGHT[p["type"]] * PLATE])


def connectors(p: dict) -> list[Connector]:
    """Connectors of a part, from its LDraw geometry (pivots measured from the official parts library)."""
    t = p["type"]
    y_axis, z_axis = np.array([0.0, 1, 0]), np.array([0.0, 0, 1])
    if t == "technic_brick":  # 1xN along x: N-1 holes across it (along y), 5.6 mm up, at the stud edges
        kind = "axle_hole" if p.get("hole") == "axle" else "round_hole"
        return [Connector(kind, np.array([(p["x"] + i) * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT]),
                          y_axis, (0.0, STUD)) for i in range(1, p["w"])]
    if t == "axle":
        return [Connector("axle", np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT]), y_axis,
                          (0.0, p["length"] * STUD))]
    if t == "wheel":
        return [Connector("hub", np.array([p["x"] * STUD, p["y"] * STUD, p["z"] * PLATE + HOLE_HEIGHT]), y_axis,
                          (0.0, wheel_studs(p.get("size", DEFAULT_WHEEL)) * STUD))]
    if t in ("hinge_base", "hinge_top"):  # swivel hinge 2429/2430: pin vertical, on the long edge at the joint
        pin = np.array([p["pin_x"] * STUD, p["pin_y"] * STUD, p["z"] * PLATE])
        return [Connector("hinge_pin_a" if t == "hinge_base" else "hinge_pin_b", pin, z_axis, (0.0, PLATE))]
    return []


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
    return min(s1, t1) - max(s0, t0) > -TOL


JOINT = {  # (kind, kind) -> joint type
    frozenset(["axle", "round_hole"]): "revolute",
    frozenset(["axle", "axle_hole"]): "rigid",
    frozenset(["axle", "hub"]): "rigid",
    frozenset(["hinge_pin_a", "hinge_pin_b"]): "revolute",
}


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
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = parts[i], parts[j]
            if pi["type"] in HEIGHT and pj["type"] in HEIGHT and _studs_rigid(pi, pj):
                links.append((i, j, "rigid", None, None))
            for a in cons[i]:
                for b in cons[j]:
                    kind = JOINT.get(frozenset([a.kind, b.kind]))
                    if kind and _on_axis(a, b):
                        links.append((i, j, kind, a.point, a.axis))
    for i, j, kind, _, _ in links:
        if kind == "rigid":
            parent[find(i)] = find(j)
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
    fixed = [p for k, p in enumerate(parts) if k not in moving and k not in ignore and p["type"] in HEIGHT]
    pts = np.vstack([_corners(parts[k]) for k in moving if parts[k]["type"] in HEIGHT and k not in ignore])
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    for s in range(1, steps + 1):
        th = np.radians(degrees * s / steps)
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        moved = (pts - point) @ R.T + point
        if any(_inside(moved, f) for f in fixed):
            return degrees * s / steps
    return None
