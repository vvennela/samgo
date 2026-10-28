"""Functional parts and the function verifier (S3c): does a vehicle build actually roll?

Structure stays on the stud grid (studs in x/y, plates in z). Functional geometry is in
millimeters, because Technic holes sit between studs: a 1xL Technic brick has L-1 holes at the
stud edges along its length, 5.8 mm above its bottom, with the hole axis horizontal and across
the brick. Parts, as the model writes them:

    {"type": "technic_brick", "w": 1, "l": 8, "x": 0, "y": 1, "z": 0, "color": "black", "hole": "round"}
    {"type": "axle", "axis": "y", "x": 1, "z": 0, "y": -1, "length": 8}   # x = stud edge, z = brick bottom
    {"type": "wheel", "axis": "y", "x": 1, "z": 0, "y": -1, "width": 2}   # on that axle, studs y..y+width

An axle with axis "y" lies on the stud edge x (between columns x-1 and x) at the hole height of
bricks whose bottom is at plate z, spanning y..y+length; axis "x" swaps the roles of x and y.
Dimensions are standard LEGO Technic; they are constants here so they can be changed in one place.
"""

from dataclasses import dataclass

from samgo.verifier import Issue, Report

STUD, PLATE = 8.0, 3.2           # mm
HOLE_HEIGHT = 5.8                # hole axis above the bottom of a Technic brick, mm
AXLE_RADIUS = 2.4
WHEEL_DIAMETER = 30.4            # LEGO 30.4 x 14 wheel with tire
TECHNIC_LENGTHS = (2, 4, 6, 8)
TOLERANCE = 0.3                  # mm; contact closer than this is not an intersection
GROUND_CLEARANCE = 1.0           # wheels must reach this far below the lowest body part, mm

FUNCTIONAL = {"technic_brick", "axle", "wheel"}


@dataclass(frozen=True)
class Box:  # an axis-aligned part volume in mm
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float


def structural(parts: list[dict]) -> list[dict]:
    """The pieces the structural verifier (S3a) sees: plates, bricks, and Technic bricks as plain bricks.

    Axles live inside holes and wheels outside the grid, so neither occupies grid cells.
    """
    out = []
    for p in parts:
        if not isinstance(p, dict) or p.get("type") in ("axle", "wheel"):
            continue
        if p.get("type") == "technic_brick":
            p = {k: v for k, v in p.items() if k != "hole"} | {"type": "brick"}
        out.append(p)
    return out


def _box(p: dict) -> Box:
    h = 3 if p["type"] in ("brick", "technic_brick") else 1
    return Box(p["x"] * STUD, (p["x"] + p["w"]) * STUD, p["y"] * STUD, (p["y"] + p["l"]) * STUD,
               p["z"] * PLATE, (p["z"] + h) * PLATE)


def _holes(p: dict) -> list[tuple[str, float, float, float, float]]:
    """(axis, edge mm, z mm, span start, span end) for each hole of a Technic brick."""
    z = p["z"] * PLATE + HOLE_HEIGHT
    if p["w"] == 1:   # long in y: holes on y edges, axis x
        return [("x", (p["y"] + i) * STUD, z, p["x"] * STUD, (p["x"] + 1) * STUD) for i in range(1, p["l"])]
    return [("y", (p["x"] + i) * STUD, z, p["y"] * STUD, (p["y"] + 1) * STUD) for i in range(1, p["w"])]


def _line(a: dict) -> tuple[str, float, float, float, float]:
    """(axis, edge mm, z mm, span start mm, span end mm) of an axle or wheel."""
    edge, start = (a["x"], a["y"]) if a["axis"] == "y" else (a["y"], a["x"])
    size = a.get("length", a.get("width"))
    return a["axis"], edge * STUD, a["z"] * PLATE + HOLE_HEIGHT, start * STUD, (start + size) * STUD


def _circle_hits_box(axis: str, edge: float, z: float, r: float, s0: float, s1: float, b: Box) -> bool:
    """Does a cylinder (radius r, along `axis`, from s0 to s1) intersect box b by more than TOLERANCE?"""
    along = (b.y0, b.y1) if axis == "y" else (b.x0, b.x1)
    across = (b.x0, b.x1) if axis == "y" else (b.y0, b.y1)
    if min(s1, along[1]) - max(s0, along[0]) <= TOLERANCE:
        return False
    dx = max(across[0] - edge, 0, edge - across[1])
    dz = max(b.z0 - z, 0, z - b.z1)
    return dx * dx + dz * dz < (r - TOLERANCE) ** 2


def _validate(p: dict) -> str | None:
    t = p.get("type")
    try:
        if t == "technic_brick":
            if sorted((p["w"], p["l"]))[0] != 1 or max(p["w"], p["l"]) not in TECHNIC_LENGTHS:
                return f"no {p['w']}x{p['l']} Technic brick (1x2, 1x4, 1x6, 1x8)"
            if p.get("hole", "round") not in ("round", "axle"):
                return "hole must be round or axle"
        elif t == "axle":
            if p["axis"] not in ("x", "y") or not 2 <= int(p["length"]) <= 16:
                return "axle needs axis x or y and length 2-16"
        elif t == "wheel":
            if p["axis"] not in ("x", "y") or int(p.get("width", 2)) != 2:
                return "wheel needs axis x or y and width 2"
        for k in ("x", "y", "z"):
            int(p[k])
    except (KeyError, TypeError, ValueError):
        return "missing or bad fields"
    return None


def verify_function(parts: list[dict]) -> Report:
    """S3c for a rolling vehicle: axles turn in aligned round holes, wheels sit on axles and clear
    everything, wheels are the lowest contact, and there are at least two parallel wheeled axles."""
    issues: list[Issue] = []
    for i, p in enumerate(parts):
        if isinstance(p, dict) and p.get("type") in FUNCTIONAL and (err := _validate(p)):
            issues.append(Issue("invalid_part", f"part {i}: {err}", [i]))
    if issues:
        return Report(False, issues, {})
    technic = [(i, p) for i, p in enumerate(parts) if isinstance(p, dict) and p.get("type") == "technic_brick"]
    axles = [(i, p) for i, p in enumerate(parts) if isinstance(p, dict) and p.get("type") == "axle"]
    wheels = [(i, p | {"width": p.get("width", 2)}) for i, p in enumerate(parts)
              if isinstance(p, dict) and p.get("type") == "wheel"]
    solids = [(i, p, _box(p)) for i, p in enumerate(parts)
              if isinstance(p, dict) and p.get("type") in ("brick", "plate", "technic_brick")]

    wheeled = []  # (axle index, axis) for axles that turn and carry wheels
    for ai, a in axles:
        axis, edge, z, s0, s1 = _line(a)
        through = []  # Technic bricks whose hole this axle passes through
        for ti, t in technic:
            for h_axis, h_edge, h_z, h0, h1 in _holes(t):
                if h_axis == axis and abs(h_edge - edge) < 0.01 and abs(h_z - z) < 0.01 and h0 >= s0 - 0.01 \
                        and h1 <= s1 + 0.01:
                    through.append(ti)
        if len(set(through)) < 2:
            issues.append(Issue("axle_unsupported", f"axle {ai} passes through {len(set(through))} aligned "
                                                    "Technic hole(s); it needs 2", [ai]))
            continue
        if locked := [ti for ti in set(through) if parts[ti].get("hole") == "axle"]:
            issues.append(Issue("axle_locked", f"axle {ai} runs through an axle-shaped hole, so it cannot turn",
                                [ai, *locked]))
        if blocked := [si for si, _, b in solids if si not in through
                       and _circle_hits_box(axis, edge, z, AXLE_RADIUS, s0, s1, b)]:
            issues.append(Issue("axle_blocked", f"axle {ai} passes through solid parts", [ai, *blocked]))
        mine = [wi for wi, w in wheels if _line(w)[:3] == (axis, edge, z)
                and _line(w)[3] >= s0 - 0.01 and _line(w)[4] <= s1 + 0.01]
        if mine and not locked:
            wheeled.append((ai, axis, len(mine)))

    r = WHEEL_DIAMETER / 2
    for wi, w in wheels:
        axis, edge, z, s0, s1 = _line(w)
        on_axle = any(_line(a)[:3] == (axis, edge, z) and _line(a)[3] <= s0 + 0.01 and _line(a)[4] >= s1 - 0.01
                      for _, a in axles)
        if not on_axle:
            issues.append(Issue("wheel_off_axle", f"wheel {wi} is not on an axle", [wi]))
            continue
        if hit := [si for si, _, b in solids if _circle_hits_box(axis, edge, z, r, s0, s1, b)]:
            issues.append(Issue("wheel_blocked", f"wheel {wi} would hit parts when it turns", [wi, *hit]))

    if len(wheeled) < 2 or any(n < 2 for _, _, n in wheeled):
        issues.append(Issue("too_few_wheels", f"{len(wheeled)} turning axle(s) with wheels; a car needs 2 axles "
                                              "with a wheel on each side"))
    elif len({axis for _, axis, _ in wheeled}) > 1:
        issues.append(Issue("axles_misaligned", "wheeled axles are not parallel, so it cannot roll straight",
                            [ai for ai, _, _ in wheeled]))

    if wheels and solids:
        wheel_bottom = min(_line(w)[2] - r for _, w in wheels)
        body_bottom = min(b.z0 for _, _, b in solids)
        clearance = body_bottom - wheel_bottom
        if clearance < GROUND_CLEARANCE:
            issues.append(Issue("body_drags", f"the body sits {clearance:.1f} mm above the wheels' contact; it "
                                              f"needs {GROUND_CLEARANCE} mm, so the wheels must be the lowest point"))
    metrics = {"turning_axles": len(wheeled), "wheels": len(wheels)}
    return Report(not issues, issues, metrics)
