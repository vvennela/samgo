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

Technic connection parts follow the axle's convention (a line along `axis` at Technic hole height, from y to
y+length for axis "y"); x, y, z may be half steps (0.5) since pins and blocks sit between grid lines:

    {"type": "pin", "axis": "y", "x": 1, "y": 0, "z": 0, "length": 2, "friction": True}   # 2780; False: 3673; 3: 6558
    {"type": "axle_pin", "axis": "y", "x": 1, "y": 0, "z": 0, "axle_end": "end"}          # 43093, pin at y..y+1
    {"type": "axle_joiner", "axis": "y", "x": 1, "y": 2, "z": 0}                          # 6538a, 2 long
    {"type": "half_bush", "axis": "y", "x": 1, "y": 2, "z": 0}                            # 32123a, half a stud
    {"type": "cross_block", "axis": "y", "x": 1, "y": 2, "z": 0, "pin_at": "+z"}          # 6536, pin hole 1 stud up
    {"type": "liftarm", "axis": "x", "x": 1, "y": 0, "z": 0, "length": 5}                 # 32316

A liftarm (thick beam) lies along `axis` with `length` round holes, one on each stud edge x..x+length-1 (axis x;
for axis y swap x and y), across it in the row y..y+1, at the hole height of a Technic brick at plate z: pinned to the side of a Technic brick, its
holes line up with the brick's. A pin with axis "z" stands upright at stud point (x, y), from plate z up.
Dimensions are standard LEGO Technic; they are constants here so they can be changed in one place.
"""

from dataclasses import dataclass

from samgo.verifier import Issue, Report

STUD, PLATE = 8.0, 3.2           # mm
HOLE_HEIGHT = 5.6                # hole axis above the bottom of a Technic brick: 14 LDU in the LDraw parts
AXLE_RADIUS = 2.4
TECHNIC_LENGTHS = (2, 4, 6, 8)
# Real Technic axle lengths (studs) and the ones made with a stop at one end (LDraw 3704...3708, 50451, 50450;
# stop axles 24316, 87083, 15462, 55013).
AXLE_LENGTHS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32)
STOP_AXLE_LENGTHS = (3, 4, 5, 8)
# Wheel-with-tyre sizes measured from the LDraw tyres: name -> (outer diameter mm, tyre width mm, LDraw tyre, rim).
WHEELS = {
    "30.4x14": (30.4, 14, "6578", "6014"),
    "43.2x22": (43.2, 22, "44309", "56145"),
    "56x28": (56.0, 28, "41897", "41896"),
    "62.4x20": (62.4, 20, "32019", "86652"),
    "68.8x36": (68.8, 36, "44771", "44772"),
    "81.6x38": (81.6, 38, "56907", "56908"),
}
DEFAULT_WHEEL = "30.4x14"
WHEEL_DIAMETER = WHEELS[DEFAULT_WHEEL][0]


def wheel_studs(size: str) -> int:
    """Studs of axle a wheel of this size takes up (its tyre width, rounded up)."""
    return -(-int(WHEELS[size][1]) // int(STUD))
TOLERANCE = 0.3                  # mm; contact closer than this is not an intersection
GROUND_CLEARANCE = 1.0           # wheels must reach this far below the lowest body part, mm

FUNCTIONAL = {"technic_brick", "axle", "wheel", "bush", "pin", "axle_pin", "axle_joiner", "half_bush", "cross_block",
              "liftarm"}
# Technic bush (LDraw 3713): a 1-stud collar that grips an axle, used to hold a wheel in place along it.

# Technic connection parts, measured from the official LDraw library (20 LDU = 1 stud = 8 mm). Pins: a 1-stud
# shaft each side of the collar (2780, 3673); 6558 is 3 long (1 + 2). 43093: a 1-stud pin and a 1-stud axle end.
# 6538a joins two axles in line in a 2-stud tube of axle holes. 32123a: axle hole, 10 LDU (half a stud) long.
# 6536: 1x1x2 block, an axle hole and a pin hole 20 LDU apart with perpendicular axes. Thick liftarms: 20 LDU
# thick, 18 LDU (7.2 mm) tall, round holes 20 LDU apart across the beam, as long as their hole count.
LIFTARM_LENGTHS = (3, 5, 7, 9, 11, 15)
TECHNIC_LDRAW = {("pin", 2, True): "2780", ("pin", 2, False): "3673", ("pin", 3, True): "6558",
                 "axle_pin": "43093", "axle_joiner": "6538a", "half_bush": "32123a", "cross_block": "6536",
                 ("liftarm", 3): "32523", ("liftarm", 5): "32316", ("liftarm", 7): "32524", ("liftarm", 9): "40490",
                 ("liftarm", 11): "32525", ("liftarm", 15): "32278", ("liftarm", 2): "60483"}
# Thick liftarms with an axle hole, so they lock onto an axle (a crank arm): length -> LDraw. 60483 is a 1x2 thick
# beam with a pin hole and an axle hole; "axle_hole": "start" or "end" says which end hole is the axle hole.
AXLE_HOLE_LIFTARMS = {2: "60483"}
PIN_RADIUS = 2.4                 # pin shaft, 6 LDU
TUBE_RADIUS = 3.6                # outside of joiners, bushes and the liftarm's rounded ends, 9 LDU
BEAM_HEIGHT = 7.2                # thick liftarm, 18 LDU
LENGTH = {"axle_pin": 2, "axle_joiner": 2, "half_bush": 0.5, "cross_block": 1}  # studs along the axis
# Parts that live in holes or on axles and hold by pins, not studs: never stud-grid cells.
OFF_GRID = {"axle", "wheel", "bush", "pin", "axle_pin", "axle_joiner", "half_bush", "cross_block", "liftarm"}


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

    Axles live inside holes and wheels outside the grid, so neither occupies grid cells; pins, liftarms and the
    other Technic connectors hold by pins, not studs, so they are left out too (kinematics.connected links them).
    """
    out = []
    for p in parts:
        if not isinstance(p, dict) or p.get("type") in OFF_GRID:
            continue
        if p.get("type") == "technic_brick":
            p = {k: v for k, v in p.items() if k != "hole"} | {"type": "brick"}
        out.append(p)
    return out


def extent(p: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(low corner, high corner) in mm of a liftarm or cross block, the Technic connectors with a body."""
    h = p["z"] * PLATE + HOLE_HEIGHT
    if p["type"] == "liftarm":  # rounded ends reach half a stud past the end holes
        along, row = (p["x"], p["y"]) if p["axis"] == "x" else (p["y"], p["x"])
        a0, a1, b0, b1 = (along - 0.5) * STUD, (along + p["length"] - 0.5) * STUD, row * STUD, (row + 1) * STUD
        x0, x1, y0, y1 = (a0, a1, b0, b1) if p["axis"] == "x" else (b0, b1, a0, a1)
        return (x0, y0, h - BEAM_HEIGHT / 2), (x1, y1, h + BEAM_HEIGHT / 2)
    # cross block: a 1-stud cube round the axle hole, and one round the pin hole beside it
    e, s = (p["x"], p["y"]) if p["axis"] == "y" else (p["y"], p["x"])
    c = [e * STUD, (s + 0.5) * STUD, h] if p["axis"] == "y" else [(s + 0.5) * STUD, e * STUD, h]
    k = "xyz".index(p.get("pin_at", "+z")[1])
    d = STUD if p.get("pin_at", "+z")[0] == "+" else -STUD
    lo, hi = [v - STUD / 2 for v in c], [v + STUD / 2 for v in c]
    lo[k], hi[k] = min(lo[k], lo[k] + d), max(hi[k], hi[k] + d)
    return tuple(lo), tuple(hi)


def _box(p: dict) -> Box:
    if p["type"] in ("liftarm", "cross_block"):
        lo, hi = extent(p)
        return Box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    h = 3 if p["type"] in ("brick", "technic_brick") else 1
    return Box(p["x"] * STUD, (p["x"] + p["w"]) * STUD, p["y"] * STUD, (p["y"] + p["l"]) * STUD,
               p["z"] * PLATE, (p["z"] + h) * PLATE)


def _holes(p: dict) -> list[tuple[str, float, float, float, float]]:
    """(axis, edge mm, z mm, span start, span end) for each hole of a Technic brick."""
    z = p["z"] * PLATE + HOLE_HEIGHT
    if p["w"] == 1:   # long in y: holes on y edges, axis x
        return [("x", (p["y"] + i) * STUD, z, p["x"] * STUD, (p["x"] + 1) * STUD) for i in range(1, p["l"])]
    return [("y", (p["x"] + i) * STUD, z, p["y"] * STUD, (p["y"] + 1) * STUD) for i in range(1, p["w"])]


def _beam_holes(p: dict) -> list[tuple[str, float, float, float, float]]:
    """The same for a liftarm: a round hole across it on each stud edge along its length."""
    z = p["z"] * PLATE + HOLE_HEIGHT
    across = "y" if p["axis"] == "x" else "x"
    along, row = (p["x"], p["y"]) if p["axis"] == "x" else (p["y"], p["x"])
    return [(across, (along + i) * STUD, z, row * STUD, (row + 1) * STUD) for i in range(p["length"])]


def _round_holes(p: dict) -> list[tuple[str, float, float, float, float]]:
    return _holes(p) if p["type"] == "technic_brick" else _beam_holes(p)


def _line(a: dict) -> tuple[str, float, float, float, float]:
    """(axis, edge mm, z mm, span start mm, span end mm) of an axle, wheel, bush or half bush."""
    edge, start = (a["x"], a["y"]) if a["axis"] == "y" else (a["y"], a["x"])
    size = a.get("length", a.get("width", 1 if a.get("type") == "bush" else LENGTH.get(a.get("type"))))
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
            lengths = STOP_AXLE_LENGTHS if p.get("stop") else AXLE_LENGTHS
            if p["axis"] not in ("x", "y") or int(p["length"]) not in lengths:
                kind = "stop axles come" if p.get("stop") else "axles come"
                return f"axle needs axis x or y; {kind} in lengths {', '.join(map(str, lengths))}"
        elif t == "bush":
            if p["axis"] not in ("x", "y") or int(p.get("width", 1)) != 1:
                return "bush needs axis x or y and is 1 stud wide"
        elif t == "wheel":
            size = p.get("size", DEFAULT_WHEEL)
            if p["axis"] not in ("x", "y") or size not in WHEELS:
                return f"wheel needs axis x or y and a size from {', '.join(WHEELS)}"
            if int(p.get("width", wheel_studs(size))) != wheel_studs(size):
                return f"a {size} wheel is {wheel_studs(size)} studs wide"
        elif t in ("pin", "axle_pin"):
            if p["axis"] not in ("x", "y", "z"):
                return f"{t} needs axis x, y or z"
            if t == "pin" and (int(p.get("length", 2)), bool(p.get("friction", True))) not in \
                    ((2, True), (2, False), (3, True)):
                return "pins are 2 long (with friction 2780, without 3673) or 3 long with friction (6558)"
            if t == "axle_pin" and p.get("axle_end", "end") not in ("start", "end"):
                return "axle_end must be start or end"
        elif t in ("axle_joiner", "half_bush", "cross_block", "liftarm"):
            if p["axis"] not in ("x", "y"):
                return f"{t} needs axis x or y"
            if t == "liftarm" and "axle_hole" in p:
                if int(p["length"]) not in AXLE_HOLE_LIFTARMS or p["axle_hole"] not in ("start", "end"):
                    return "a liftarm with an axle hole is 2 long (60483), axle_hole start or end"
            elif t == "liftarm" and int(p["length"]) not in LIFTARM_LENGTHS:
                return f"liftarms come with {', '.join(map(str, LIFTARM_LENGTHS))} holes"
            if t == "cross_block" and p.get("pin_at", "+z") not in [s + k for s in "+-" for k in "xyz" if k != p["axis"]]:
                return "pin_at must be +/- one of the two directions across the axle hole (e.g. +z)"
        for k in ("x", "y", "z"):
            if t in OFF_GRID - {"axle", "wheel", "bush"}:  # connectors may sit half a step off the grid
                if float(p[k]) * 2 != int(float(p[k]) * 2):
                    return f"{k} must be a whole or half step"
            else:
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
    # Technic bricks and liftarms: the parts with round holes an axle can turn in.
    technic = [(i, p) for i, p in enumerate(parts) if isinstance(p, dict) and p.get("type") in ("technic_brick", "liftarm")]
    axles = [(i, p) for i, p in enumerate(parts) if isinstance(p, dict) and p.get("type") == "axle"]
    wheels = [(i, p | {"width": wheel_studs(p.get("size", DEFAULT_WHEEL))}) for i, p in enumerate(parts)
              if isinstance(p, dict) and p.get("type") == "wheel"]
    solids = [(i, p, _box(p)) for i, p in enumerate(parts)
              if isinstance(p, dict) and p.get("type") in ("brick", "plate", "technic_brick", "liftarm", "cross_block")]

    wheeled = []  # (axle index, axis) for axles that turn and carry wheels
    for ai, a in axles:
        axis, edge, z, s0, s1 = _line(a)
        through = []  # Technic bricks whose hole this axle passes through
        for ti, t in technic:
            for h_axis, h_edge, h_z, h0, h1 in _round_holes(t):
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
        # A wheel on each end clamps an axle in place; an axle carrying one wheel (a stub) needs a stop at its end,
        # or it slides out of the holes.
        if len(mine) == 1 and not a.get("stop"):
            issues.append(Issue("axle_loose", f"axle {ai} carries one wheel and has no stop, so it slides out; use a "
                                              f"stop axle ({', '.join(map(str, STOP_AXLE_LENGTHS))} long) or wheels "
                                              "on both ends", [ai]))
        if mine and not locked:
            wheeled.append((ai, axis, (axis, round(edge, 2), round(z, 2)), len(mine)))

    # Each wheel must be held on both sides along its axle: by a beam's hole, a bush, another wheel, or the axle
    # ending there. A gap lets it slide along the axle (real builds fill it with bushes).
    bushes = [p for p in parts if isinstance(p, dict) and p.get("type") in ("bush", "half_bush")]
    for wi, w in wheels:
        axis, edge, z, s0, s1 = _line(w)
        line_axles = [_line(a) for _, a in axles if _line(a)[:3] == (axis, edge, z)]
        stops = [(l[3], l[4]) for l in line_axles]
        neighbours = [(_line(b)[3], _line(b)[4]) for b in bushes if _line(b)[:3] == (axis, edge, z)]
        neighbours += [(_line(o)[3], _line(o)[4]) for oi, o in wheels if oi != wi and _line(o)[:3] == (axis, edge, z)]
        for _, t in technic:
            neighbours += [(h0, h1) for h_axis, h_edge, h_z, h0, h1 in _round_holes(t)
                           if h_axis == axis and abs(h_edge - edge) < 0.01 and abs(h_z - z) < 0.01]
        for face, side in ((s0, "start"), (s1, "end")):
            ends_here = any(abs((a0 if side == "start" else a1) - face) < 0.01 or
                            ((a0 > face - 0.01) if side == "start" else (a1 < face + 0.01)) for a0, a1 in stops)
            held = any(abs((n1 if side == "start" else n0) - face) < 0.01 for n0, n1 in neighbours)
            if line_axles and not (ends_here or held):
                issues.append(Issue("wheel_slides", f"wheel {wi} can slide along its axle: nothing holds its "
                                                    "one side (add a bush)", [wi]))
                break

    for wi, w in wheels:
        r = WHEELS[w.get("size", DEFAULT_WHEEL)][0] / 2
        axis, edge, z, s0, s1 = _line(w)
        on_axle = any(_line(a)[:3] == (axis, edge, z) and _line(a)[3] <= s0 + 0.01 and _line(a)[4] >= s1 - 0.01
                      for _, a in axles)
        if not on_axle:
            issues.append(Issue("wheel_off_axle", f"wheel {wi} is not on an axle", [wi]))
            continue
        if hit := [si for si, _, b in solids if _circle_hits_box(axis, edge, z, r, s0, s1, b)]:
            issues.append(Issue("wheel_blocked", f"wheel {wi} would hit parts when it turns", [wi, *hit]))

    # Axles on one line (a through axle, or two stub axles facing each other) form one wheel pair.
    lines: dict[tuple, int] = {}
    for _, _, line, n in wheeled:
        lines[line] = lines.get(line, 0) + n
    if len(lines) < 2 or any(n < 2 for n in lines.values()):
        issues.append(Issue("too_few_wheels", f"{len(lines)} turning axle line(s) with wheels; a car needs 2 axle "
                                              "lines with a wheel on each side"))
    elif len({line[0] for line in lines}) > 1:
        issues.append(Issue("axles_misaligned", "wheeled axles are not parallel, so it cannot roll straight",
                            [ai for ai, _, _, _ in wheeled]))

    from .kinematics import unseated  # kinematics builds on this module

    for i, why in unseated(parts):
        issues.append(Issue("connector_loose", f"part {i} ({parts[i]['type']}) {why}", [i]))

    if wheels and solids:
        wheel_bottom = min(_line(w)[2] - WHEELS[w.get("size", DEFAULT_WHEEL)][0] / 2 for _, w in wheels)
        body_bottom = min(b.z0 for _, _, b in solids)
        clearance = body_bottom - wheel_bottom
        if clearance < GROUND_CLEARANCE:
            issues.append(Issue("body_drags", f"the body sits {clearance:.1f} mm above the wheels' contact; it "
                                              f"needs {GROUND_CLEARANCE} mm, so the wheels must be the lowest point"))
    metrics = {"turning_axles": len(wheeled), "wheels": len(wheels)}
    return Report(not issues, issues, metrics)
