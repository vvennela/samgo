"""The build animation's order and paths, with every move checked collision-free (formal/AnimationPath.lean).

Order: a model that rolls on wheels is built the way people build one - the rolling chassis first (Technic bricks,
then the axles slid through their holes, then bushes and wheels), standing on its wheels, then the body bottom-up
(in the certified order when there is one). Anything else: the certified order, else layers bottom-up.

Paths: each part arrives along one straight direction: plates and bricks drop from above (+z); axles, bushes and
wheels slide in along their own axis (either way). A direction is used only if the move is clear of every part
already placed, by the exact condition proved in formal/AnimationPath.lean:
- `slide_clear`: for each placed box, either its projection on the two other axes is disjoint from the moving
  box's, or it lies entirely behind the moving box along the direction - then no translation by t >= 0 overlaps it;
- `hole_clear`: an axle sliding along the line of a Technic brick's round hole stays inside the hole's channel, so it
  never meets the brick's plastic.
Boxes are in exact integer units of 0.2 mm (every LEGO dimension used here is a multiple: stud 8 mm = 40, plate
3.2 mm = 16, hole axis 5.6 mm = 28, axle half-width 2.4 mm = 12).
"""

from fractions import Fraction

from .kinematics import box
from .technic import HOLE_HEIGHT, PLATE, STUD, _holes, _line

UNIT = Fraction(1, 5)  # mm
AXLE_HALF = 12          # 2.4 mm: an axle's cross is inside a square of half-width 2.4 mm
HOLE_HALF = 12          # a round hole of radius 2.4 mm (the axle fits it exactly; the cross turns inside it)
ON_AXIS = {"axle", "wheel", "bush", "half_bush"}
CHASSIS = ("technic_brick", "axle", "bush", "half_bush", "wheel")


def _u(v) -> int:
    q = Fraction(v).limit_denominator(1000) / UNIT
    if q.denominator != 1:
        raise ValueError(f"{v} mm is not a multiple of 0.2 mm")
    return int(q)


def ibox(p: dict) -> tuple[list[int], list[int]]:
    """The part's box in 0.2 mm units: (lo[x, y, z], hi[x, y, z])."""
    if p["type"] in ("axle", "bush", "half_bush"):
        axis, edge, z, s0, s1 = _line(p)
        lo, hi = [0, 0, 0], [0, 0, 0]
        a = 0 if axis == "x" else 1
        b = 1 - a
        lo[a], hi[a] = _u(s0), _u(s1)
        lo[b], hi[b] = _u(edge) - AXLE_HALF, _u(edge) + AXLE_HALF
        lo[2], hi[2] = _u(z) - AXLE_HALF, _u(z) + AXLE_HALF
        return lo, hi
    lo, hi = box(p)
    return [_u(round(v, 3)) for v in lo], [_u(round(v, 3)) for v in hi]


def slide_clear(moving: tuple, placed: tuple, axis: int, sign: int) -> bool:
    """formal/AnimationPath.lean `slide_clear`: moving the box along `axis` by sign * t (t >= 0, towards its final
    place) never overlaps `placed` - the projections on the other axes are disjoint (open intervals), or `placed`
    is entirely behind it."""
    (mlo, mhi), (plo, phi) = moving, placed
    others = [k for k in range(3) if k != axis]
    if any(mhi[k] <= plo[k] or phi[k] <= mlo[k] for k in others):
        return True
    return phi[axis] <= mlo[axis] if sign > 0 else mhi[axis] <= plo[axis]


def hole_clear(axle: dict, brick: dict) -> bool:
    """formal/AnimationPath.lean `hole_clear`: the axle runs along one of the brick's round holes (same line), so
    sliding along it keeps it inside the hole's channel."""
    axis, edge, z, _, _ = _line(axle)
    return any(h[0] == axis and _u(h[1]) == _u(edge) and _u(h[2]) == _u(z) for h in _holes(brick))


def on_axle(p: dict, axle: dict) -> bool:
    """`p` sits on this axle through a hole of its own: a wheel or bush on the same line, or a liftarm with a hole on
    it (formal/AnimationPath.lean `hole_clear`, its hub or arm hole as the hole)."""
    axis, edge, z, _, _ = _line(axle)
    if p["type"] in ("wheel", "bush", "half_bush"):
        if p.get("axis") != axle.get("axis"):
            return False
        a = _line(p)
        return _u(a[1]) == _u(edge) and _u(a[2]) == _u(z)
    if p["type"] == "liftarm":
        from .technic import _beam_holes

        return any(h[0] == axis and _u(h[1]) == _u(edge) and _u(h[2]) == _u(z) for h in _beam_holes(p))
    return False


def through_solid(parts: list[dict]) -> list[tuple[int, int]]:
    """(axle or bush, part) pairs where the axle line passes through a solid part anywhere but one of its round holes
    or a hub on it: impossible to build (the axle would have to pass through plastic)."""
    from .kinematics import HEIGHT

    out = []
    boxes = {}
    for i, a in enumerate(parts):
        if a["type"] not in ("axle", "bush", "half_bush"):
            continue
        lo, hi = boxes.setdefault(i, ibox(a))
        line = a if a["type"] == "axle" else None
        for j, q in enumerate(parts):
            if j == i or q["type"] not in HEIGHT:
                continue
            if q["type"] == "technic_brick" and line is not None and hole_clear(a, q):
                continue
            qlo, qhi = boxes.setdefault(j, ibox(q))
            if all(lo[k] < qhi[k] and qlo[k] < hi[k] for k in range(3)):
                out.append((i, j))
    return out


def order(parts: list[dict], manual: dict | None = None) -> list[int]:
    """The step of each part (see the module docstring)."""
    base = [0] * len(parts)
    if manual and manual.get("passed") and manual.get("steps"):
        for k, s in enumerate(manual["steps"]):
            for i in s["parts"]:
                base[i] = k
    else:
        zs = sorted({p.get("z", 0) for p in parts})
        base = [zs.index(p.get("z", 0)) for p in parts]
    if not any(p["type"] == "wheel" for p in parts):
        return base
    # On its wheels the body is built up from them: layers bottom-up (a manual may build it upside down).
    zs = sorted({p.get("z", 0) for p in parts})
    base = [zs.index(p.get("z", 0)) for p in parts]
    rank = {t: n for n, t in enumerate(CHASSIS)}
    axles = [p for p in parts if p["type"] == "axle"]

    bricks = [p for p in parts if p["type"] == "technic_brick"]

    def along(p, a):  # p's centre along axle a's line, mm
        axis = a.get("axis")
        if p["type"] == "liftarm":
            return (p["x"] if axis == "x" else p["y"]) * STUD + STUD / 2
        if p["type"] == "technic_brick":
            return (p["x"] + p["w"] / 2) * STUD if axis == "x" else (p["y"] + p["l"] / 2) * STUD
        _, _, _, h0, h1 = _line(p)
        return (h0 + h1) / 2

    def key(i):
        p = parts[i]
        if p["type"] in ("wheel", "bush", "half_bush", "liftarm"):  # on each axle from its bearing out, in turn
            a = next((x for x in axles if on_axle(p, x)), None)
            if a is not None:
                held = [along(b, a) for b in bricks if hole_clear(a, b)]
                _, _, _, s0, s1 = _line(a)
                anchor = sum(held) / len(held) if held else (s0 + s1) / 2
                return (2, abs(along(p, a) - anchor), p.get("x", 0), p.get("y", 0))
        return (rank.get(p["type"], 2) if rank.get(p["type"], 2) < 2 else 2, 0, p.get("z", 0), p.get("x", 0))

    hub = {i for i, p in enumerate(parts) if p["type"] == "liftarm" and any(on_axle(p, a) for a in axles)}
    chassis = sorted((i for i, p in enumerate(parts) if p["type"] in rank or i in hub), key=key)
    out = [0] * len(parts)
    step = 0
    for i in chassis:
        out[i] = step
        step += 1
    rest = sorted((i for i, p in enumerate(parts) if p["type"] not in rank and i not in hub), key=lambda i: base[i])
    last = None
    for i in rest:  # the body keeps its own grouping (a manual step, a layer) as one step
        if last is not None and base[i] != last:
            step += 1
        out[i] = step
        last = base[i]
    return out


def paths(parts: list[dict], steps: list[int]) -> tuple[list[list[int]], list[int]]:
    """For each part, the direction it arrives from (a unit vector in LEGO x, y, z: the animation moves it from
    far along that direction to its place) and the parts no clear direction exists for (should be none)."""
    boxes = [ibox(p) for p in parts]
    placed: list[int] = []
    out: list[list[int]] = [[0, 0, 1]] * len(parts)
    blocked = []
    for s in sorted(set(steps)):
        now = [i for i, st in enumerate(steps) if st == s]
        for i in now:
            p = parts[i]
            axles = [j for j in placed if parts[j]["type"] == "axle" and on_axle(p, parts[j])]
            if p["type"] in ON_AXIS or axles:
                a = 0 if (parts[axles[0]] if axles else p).get("axis") == "x" else 1
                choices = [(a, 1), (a, -1), (2, 1)]
            else:
                choices = [(2, 1), (0, 1), (0, -1), (1, 1), (1, -1), (2, -1)]
            for axis, sign in choices:
                ok = True
                for j in placed:
                    if p["type"] == "axle" and parts[j]["type"] == "technic_brick" and hole_clear(p, parts[j]):
                        continue
                    if p["type"] == "technic_brick" and parts[j]["type"] == "axle" and hole_clear(parts[j], p):
                        continue
                    if parts[j]["type"] == "axle" and on_axle(p, parts[j]) \
                            and axis == (0 if parts[j].get("axis") == "x" else 1):
                        continue  # hole_clear: a hub slides along its own axle, which stays inside its hole
                    if not slide_clear(boxes[i], boxes[j], axis, sign):
                        ok = False
                        break
                if ok:
                    v = [0, 0, 0]
                    v[axis] = sign
                    out[i] = v
                    break
            else:
                blocked.append(i)
        placed += now  # parts of one step arrive together
    return out, blocked
