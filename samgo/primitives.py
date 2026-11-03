"""Run 6 training data: generic mechanism primitives that need removing before adding, on bodies of any shape.

Two primitives, each built as a verified reference answer on a part of a body and graded by the same generic checks
as the granularity sweep (samgo.free_edit_test.check), never by anything object-specific:

  hinge  free a strip along a thin edge and hang it on clip-and-bar hinges: clear a one-stud gap through the edge
         (parts that cross it are shortened with "set", so the strip is made of the model's own parts, and what was
         cut is refilled with real plates), then a bar plate on the body and a clip plate on the strip, on top.
  leg    a raised body gets a leg under it: a brick column from its underside, two Technic bricks, an axle, two
         wheels and bushes; the wheels are the lowest parts.

Bodies: car shells (training split only) and generic thin-sheet bodies (a block with flat sheets sticking out at
random heights, sides and planforms: shelves, lids, fins, wings, ledges). The prompt shows the parts of a box
around the work plus a ring of neighbours, at one of three handoff levels (L1 a structure, L2 the edge or spot,
L3 the exact line or position). 40% of rows first give the habitual wrong answer (adding without removing),
the precise feedback for it, then the reference answer.

    python -m samgo.primitives --out data/primitives_r6 --test runs/audit/r6-primitives
"""

import argparse
import copy
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .edits import apply_edits
from .free_edit_test import RULES, check, explain
from .parts import FOOTPRINTS
from .running_gear_task import FEEDBACK

H = {"plate": 1, "brick": 3}
VALID = {tuple(sorted(f)) for f in FOOTPRINTS}
COLORS = ("white", "light_gray", "red", "blue", "yellow", "dark_gray", "tan", "black", "green", "orange")
FLAP = [{"name": "flap", "kind": "swing", "axis": "horizontal", "degrees": 20, "count": 1}]
LEG = [{"name": "leg", "kind": "spin", "count": 2, "stands": False}]
KEEP = ("structure", "connected", "flap", "leg", "rotor")


# ---------------------------------------------------------------- geometry helpers

def tile(x0: int, x1: int, y0: int, y1: int, phase: int = 0) -> list[tuple[int, int, int, int]]:
    """(x, y, w, l) rectangles of real footprints covering [x0,x1) x [y0,y1); `phase` staggers the seams."""
    out = []
    if x1 <= x0 or y1 <= y0:
        return out
    along_x = ((x1 - x0) >= (y1 - y0)) != bool(phase)  # odd layers cross-hatch the even ones
    a0, a1, b0, b1 = (x0, x1, y0, y1) if along_x else (y0, y1, x0, x1)
    b, first_band = b0, True
    while b < b1:
        bw = 2 if b1 - b >= 2 else 1
        if first_band and phase and b1 - b > 2:
            bw = 1  # a narrow first band shifts the seams across the strips too
        first_band = False
        a, first = a0, True
        while a < a1:
            room = a1 - a
            n = next(k for k in (8, 6, 4, 3, 2, 1) if k <= room)
            if first and phase and room > 2:
                n = min(n, 1 + phase)  # a short first piece shifts every seam in this strip
            first = False
            out.append((a, b, n, bw) if along_x else (b, a, bw, n))
            a += n
        b += bw
    return [r for r in out if tuple(sorted(r[2:])) in VALID]


def cells(p: dict):
    for x in range(p["x"], p["x"] + p["w"]):
        for y in range(p["y"], p["y"] + p["l"]):
            for z in range(p["z"], p["z"] + H.get(p["type"], 1)):
                yield x, y, z


def occupancy(parts: list[dict]) -> dict:
    occ = {}
    for i, p in enumerate(parts):
        if p["type"] in H or p["type"] in ("bar_plate", "clip_plate", "technic_brick"):
            for c in cells(p):
                occ[c] = i
    return occ


def swap_xy(p: dict) -> dict:
    q = dict(p)
    q["x"], q["y"] = p["y"], p["x"]
    if "w" in p:
        q["w"], q["l"] = p["l"], p["w"]
    if "facing" in p:
        q["facing"] = {"+x": "+y", "-x": "-y", "+y": "+x", "-y": "-x"}[p["facing"]]
    if "axis" in p:
        q["axis"] = {"x": "y", "y": "x"}.get(p["axis"], p["axis"])
    return q


def mirror_x(p: dict, X: int) -> dict:
    q = dict(p)
    if p["type"] in ("axle", "wheel", "bush"):
        q["x"] = X - p["x"]  # on a stud edge
    else:
        q["x"] = X - (p["x"] + p["w"])
    if "facing" in p:
        q["facing"] = {"+x": "-x", "-x": "+x"}.get(p["facing"], p["facing"])
    return q


def frame(parts: list[dict], swap: bool, mirror: bool, X: int) -> list[dict]:
    out = [swap_xy(p) if swap else dict(p) for p in parts]
    return [mirror_x(p, X) for p in out] if mirror else out


def unframe(parts: list[dict], swap: bool, mirror: bool, X: int) -> list[dict]:
    out = [mirror_x(p, X) for p in parts] if mirror else [dict(p) for p in parts]
    return [swap_xy(p) for p in out] if swap else out


# ---------------------------------------------------------------- bodies

def tile_cells(todo: set, phase: int) -> list[tuple[int, int, int, int]]:
    """(x, y, w, l) real footprints covering a set of (x, y) cells: greedy, largest first; odd phases scan and
    lay pieces the other way, so stacked layers cross each other's seams."""
    # direction before size: even layers lay pieces long along x, odd layers long along y
    along = (lambda f: f[0] >= f[1]) if not phase else (lambda f: f[1] >= f[0])
    shapes = sorted({(a, b) for a, b in VALID} | {(b, a) for a, b in VALID},
                    key=lambda f: (not along(f), -f[0] * f[1]))
    left, out = set(todo), []
    order = sorted(left, key=lambda c: (c[0], c[1]) if phase else (c[1], c[0]))
    for (x, y) in order:
        if (x, y) not in left:
            continue
        for w, l in shapes:
            block = {(x + i, y + j) for i in range(w) for j in range(l)}
            if block <= left:
                out.append((x, y, w, l))
                left -= block
                break
    return out


def sheet_body(rng: random.Random) -> list[dict]:
    """A generic body: a brick block on the ground with 1-3 flat sheets (2-3 plates thick) lying on its top and
    sticking out past its sides, each a random trapezoid (sweep, taper). Stacked layers cross-hatch."""
    from .verifier import verify

    for _ in range(20):
        bx, by = rng.randint(4, 10), rng.randint(4, 12)
        bh = rng.choice((2, 3, 4))  # bricks
        ox, oy = 14, 14
        color, sc = rng.choice(COLORS), rng.choice(COLORS)
        block = {(x, y) for x in range(ox, ox + bx) for y in range(oy, oy + by)}
        parts = [dict(type="brick", w=w, l=l, x=x, y=y, z=3 * k, color=color)
                 for k in range(bh) for (x, y, w, l) in tile_cells(block, k % 2)]
        thick = rng.choice((2, 3))
        sheet = set()
        for side in rng.sample(("+x", "-x", "+y", "-y"), rng.randint(1, 3)):
            span = by if side in ("+x", "-x") else bx
            root = rng.randint(3, span)
            reach = rng.randint(4, 10)
            sweep, taper = rng.choice((0, 0, 1, 2)), rng.choice((0, 1, 1, 2))
            u0 = rng.randint(0, span - root)
            for v in range(-3, reach):  # three rows over the block, the rest sticking out
                t = max(v, 0)
                a = u0 + (sweep * t) // max(reach - 1, 1)
                b = a + root - (taper * t) // max(reach - 1, 1)
                for u in range(a, b):
                    sheet.add({"+x": (ox + bx - 1 + v, oy + u), "-x": (ox - v, oy + u),
                               "+y": (ox + u, oy + by - 1 + v), "-y": (ox + u, oy - v)}[side])
        parts += [dict(type="plate", w=w, l=l, x=x, y=y, z=3 * bh + k, color=sc)
                  for k in range(thick) for (x, y, w, l) in tile_cells(sheet, (k + 1) % 2)]
        if min(p["x"] for p in parts) >= 0 and min(p["y"] for p in parts) >= 0 and verify(parts, None).passed:
            return parts
    raise ValueError("no connected sheet body")


def shell_body(name: str) -> list[dict]:
    """A car shell without its running gear (its Technic bricks become plain bricks): a body to edit."""
    parts = json.loads(Path(f"runs/bases_shell/{name}.json").read_text())
    return [{k: v for k, v in p.items() if k != "hole"} | {"type": "brick"} if p["type"] == "technic_brick" else p
            for p in parts if p["type"] in H or p["type"] == "technic_brick"]


def shell_names(split: str) -> list[str]:
    from .gen_running_gear import split_of

    return sorted(p.stem for p in Path("runs/bases_shell").glob("*.json") if split_of(p.stem) == split)


def body_for(rng: random.Random, split: str, sheet_share: float = 0.6) -> tuple[list[dict], str]:
    if rng.random() < sheet_share:
        return sheet_body(rng), "object"
    name = rng.choice(shell_names(split))
    return shell_body(name), "car"


# ---------------------------------------------------------------- hinge primitive

def _cut(p: dict, xg: int, y0: int, y1: int) -> list[dict]:
    """The pieces of part p left after clearing column x = xg in rows [y0, y1), split at the strip's sides:
    first the body side (x < xg in the rows, or outside the rows), then the strip side (x > xg in the rows)."""
    px0, px1, py0, py1 = p["x"], p["x"] + p["w"], p["y"], p["y"] + p["l"]
    rects = []
    if py0 < y0:
        rects.append((px0, px1, py0, min(py1, y0)))
    if py1 > y1:
        rects.append((px0, px1, max(py0, y1), py1))
    a, b = max(py0, y0), min(py1, y1)
    if a < b:
        if px0 < xg:
            rects.append((px0, min(px1, xg), a, b))
        if px1 > xg + 1:
            rects.append((max(px0, xg + 1), px1, a, b))
    out = []
    for r in rects:
        out += [dict(p, x=x, y=y, w=w, l=l) for (x, y, w, l) in tile(*r)]
    return out


def hinge_problem(body: list[dict], rng: random.Random, margin: int = 0) -> dict | None:
    """A hinge task on `body` in a random orientation, or None if no thin free edge is found there."""
    swap, mirror = rng.random() < 0.5, rng.random() < 0.5
    X = max(p["x"] + p.get("w", 1) for p in body) + 1
    parts = frame(body, swap, mirror, X)
    occ = occupancy(parts)
    if not occ:
        return None
    # the +x edge of every row: the last occupied x over all heights
    te: dict = {}
    for (x, y, z) in occ:
        te[y] = max(te.get(y, -1), x)
    rows = sorted(te)
    for _ in range(40):
        L = rng.randint(3, 7)
        if len(rows) < L:
            return None
        y0 = rng.choice(rows[: len(rows) - L + 1])
        span = list(range(y0, y0 + L))
        if any(y not in te for y in span) or max(te[y] for y in span) - min(te[y] for y in span) > 1:
            continue
        chord = rng.randint(2, 4)
        xg = min(te[y] for y in span) - chord
        gap = [(xg, y, z) for y in span for z in range(0, 60) if (xg, y, z) in occ]
        zs = sorted({z for (_, _, z) in gap})
        if not zs or len(zs) > 4 or zs[-1] - zs[0] + 1 != len(zs):
            continue  # not a thin sheet here
        # the strip and the body both continue on each side of the gap, at the same heights
        if any((x, y, z) not in occ for x in (xg - 2, xg - 1, xg + 1, xg + 2) for y in span for z in zs):
            continue
        top = zs[-1] + 1
        if any((x, y, top) in occ for y in span for x in range(xg - 2, xg + 3)):
            continue  # the hinge sits on top of the sheet
        if any((x, y, z) in occ for y in span for x in range(xg - 3, xg + 4) for z in range(top, top + 4)):
            continue  # room above the hinge line to swing
        if any((x, y, z) in occ for y in span for x in range(xg - 3, te[y] + 2) for z in range(max(zs[0] - 4, 0),
                                                                                              zs[0])):
            continue  # nothing under the strip for it to hit when it swings down
        break
    else:
        return None
    y1 = span[-1] + 1
    # the parts crossing the cut are re-laid; with a margin, also those near it (rows each side, studs inboard),
    # so the new layers interlock instead of leaving short stacks that touch nothing
    touched = sorted({occ[(x, y, z)] for (x, y, z) in occ if y0 - margin <= y < y1 + margin and x >= xg - 2 * margin})
    touched = [i for i in touched if not (parts[i]["x"] > xg and y0 <= parts[i]["y"] and
                                          parts[i]["y"] + parts[i]["l"] <= y1)]  # wholly on the strip: stays
    # what the crossing parts covered, minus the gap, re-tiled per side and layer with crossed seams
    groups: dict = {}
    for i in touched:
        p = parts[i]
        for x in range(p["x"], p["x"] + p["w"]):
            for y in range(p["y"], p["y"] + p["l"]):
                inside = y0 <= y < y1
                if inside and x == xg:
                    continue
                side = "strip" if inside and x > xg else "body"
                groups.setdefault((side, p["type"], p["z"], p["color"]), set()).add((x, y))
    strip_pieces, body_pieces = [], []
    for (side, t, z, color), cs in groups.items():
        for (x, y, w, l) in tile_cells(cs, (z // H[t]) % 2):
            (strip_pieces if side == "strip" else body_pieces).append(dict(type=t, w=w, l=l, x=x, y=y, z=z,
                                                                            color=color))
    pieces = strip_pieces + body_pieces  # crossing parts are reused for strip pieces first: the strip is the model's
    sets, adds = {}, []
    for k, i in enumerate(touched):
        sets[i] = pieces[k] if k < len(pieces) else None
    adds = pieces[len(touched):]
    hy = sorted(rng.sample(span, 2 if L >= 4 else 1))
    color = rng.choice(("dark_gray", "light_gray", "black"))
    for y in hy:
        adds.append(dict(type="bar_plate", w=2, l=1, x=xg - 2, y=y, z=top, facing="+x", color=color))
        adds.append(dict(type="clip_plate", w=2, l=1, x=xg + 1, y=y, z=top, facing="-x", color=color))
    # what the task shows: a box around the work plus a ring of neighbours
    x_lo, x_hi = xg - 6, max(te[y] for y in span) + 2
    view = [i for i, p in enumerate(parts)
            if p["x"] < x_hi and p["x"] + p["w"] > x_lo and p["y"] < y1 + 3 and p["y"] + p["l"] > y0 - 3]
    if not all(i in view for i in sets):
        return None
    return dict(kind="hinge", frame=(swap, mirror, X), parts=parts, view=view, sets=sets, adds=adds,
                where=dict(xg=xg, y0=y0, y1=y1, top=top, hinge_rows=hy, z0=zs[0], z1=zs[-1],
                           edge=max(te[y] for y in span)))


def hinge_text(prob: dict, level: str, obj: str) -> str:
    w = prob["where"]
    swap, mirror, X = prob["frame"]
    # coordinates in the instruction are the original model's: map the canonical +x edge back
    axis, other = ("y", "x") if swap else ("x", "y")
    if mirror:
        gx, outward = X - w["xg"] - 1, "-" + axis
    else:
        gx, outward = w["xg"], "+" + axis
    lo, hi = w["y0"], w["y1"] - 1
    if level == "L1":
        return (f"These are the parts of one flat structure of a{'n' if obj[0] in 'aeiou' else ''} {obj}. Add one "
                f"flap on its edge toward {outward}: free a strip of that edge and hinge it so it swings about a "
                "horizontal axis.")
    if level == "L2":
        return (f"These are the parts along one edge of a flat structure (the edge toward {outward}, rows "
                f"{other} = {lo}..{hi}). Turn the outer part of it into a flap: free it and hinge it about a "
                "horizontal axis along the edge.")
    bar = (gx - 2, "+" + axis, gx + 1, "-" + axis) if not mirror else (gx + 1, "-" + axis, gx - 2, "+" + axis)
    dims = "w=1, l=2" if swap else "w=2, l=1"
    return (f"These are the parts along one edge of a flat structure. Make a flap beyond the line {axis} = {gx} "
            f"(rows {other} = {lo}..{hi}): clear the column {axis} = {gx} in those rows and the rows {other} = "
            f"{lo - 1} and {other} = {hi + 1} beyond it, out to the edge, so the strip is free on three sides; keep "
            f"the parts beyond the line as the flap, and hinge it with bar "
            f"plates ({dims}, facing {bar[1]}, at {axis} = {bar[0]}) and clip plates ({dims}, facing {bar[3]}, at "
            f"{axis} = {bar[2]}) on top of the sheet at z = {w['top']}, in rows {other} = "
            f"{', '.join(map(str, w['hinge_rows']))}.")


# ---------------------------------------------------------------- leg primitive

def leg_problem(body: list[dict], rng: random.Random) -> dict | None:
    swap, mirror = rng.random() < 0.5, rng.random() < 0.5
    X = max(p["x"] + p.get("w", 1) for p in body) + 1
    lift = rng.choice((12, 15))  # a 30.4 mm wheel on a brick at z=3 reaches 9.5 plates up: clear of the body
    zmin = min(p["z"] for p in body)
    parts = [p | {"z": p["z"] - zmin + lift} for p in frame(body, swap, mirror, X)]
    occ = occupancy(parts)
    bottom: dict = {}
    for (x, y, z) in occ:
        bottom[(x, y)] = min(bottom.get((x, y), 99), z)
    spots = [(x, y) for (x, y), z in bottom.items() if z == lift
             and all(bottom.get((x + dx, y + dy)) == lift for dx in (0, 1) for dy in (-1, 0))]
    if not spots:
        return None
    x0, yc = rng.choice(spots)  # the column covers x0..x0+1, yc-1..yc; the axle runs along y on edge x0+1
    xg = x0 + 1
    gray = rng.choice(("light_gray", "dark_gray", "black"))
    adds = [dict(type="brick", w=2, l=2, x=x0, y=yc - 1, z=z, color=gray) for z in range(6, lift, 3)]
    adds += [dict(type="technic_brick", w=2, l=1, x=x0, y=y, z=3, color=gray, hole="round") for y in (yc - 1, yc)]
    adds += [dict(type="axle", axis="y", x=xg, z=3, y=yc - 4, length=8),
             dict(type="wheel", axis="y", x=xg, z=3, y=yc - 3, size="30.4x14", color="black"),
             dict(type="wheel", axis="y", x=xg, z=3, y=yc + 1, size="30.4x14", color="black"),
             dict(type="bush", axis="y", x=xg, z=3, y=yc - 4), dict(type="bush", axis="y", x=xg, z=3, y=yc + 3)]
    view = [i for i, p in enumerate(parts) if p["z"] <= lift + 3 and abs(p["x"] - x0) <= 8 and abs(p["y"] - yc) <= 8]
    return dict(kind="leg", frame=(swap, mirror, X), parts=parts, view=view, sets={}, adds=adds,
                where=dict(x0=x0, yc=yc, lift=lift))


def leg_text(prob: dict, level: str, obj: str) -> str:
    w = prob["where"]
    swap, mirror, X = prob["frame"]
    axis, other = ("y", "x") if swap else ("x", "y")
    x0 = (X - w["x0"] - 2) if mirror else w["x0"]
    if level == "L1":
        return (f"These are the lower parts of a{'n' if obj[0] in 'aeiou' else ''} {obj}, raised off the ground. Add "
                "one leg under it with an axle and two wheels that spin; the wheels must be the lowest parts.")
    if level == "L2":
        return (f"These are the underside parts of a raised object around {axis} = {x0}..{x0 + 1}, {other} = "
                f"{w['yc'] - 1}..{w['yc']}. Add a leg there: a column down from the underside, an axle and two "
                "wheels that spin, lowest of all.")
    dims = "w=1, l=2" if swap else "w=2, l=1"
    edge = (X - w["x0"] - 1) if mirror else w["x0"] + 1
    return (f"These are the underside parts of a raised object. Add a leg: a 2x2 column of bricks from the "
            f"underside at z = {w['lift']} down to z = 6, on two 1x2 Technic bricks ({dims}) at {axis} = {x0}, "
            f"{other} = {w['yc'] - 1} and {other} = {w['yc']}, z = 3; an axle along {other} on stud edge {axis} = "
            f"{edge}, z = 3, from {other} = {w['yc'] - 4}, length 8; wheels at {other} = {w['yc'] - 3} and "
            f"{other} = {w['yc'] + 1}; bushes at {other} = {w['yc'] - 4} and {other} = {w['yc'] + 3}.")


# ---------------------------------------------------------------- cases and rows

def case(seed: int, kind: str, split: str = "train") -> dict | None:
    """The smallest verified answer: for a hinge, the bare cut first, then re-laying a margin around it."""
    for margin in ((0,) if kind == "hinge" else (0,)):  # with clear, the code re-lays the cut: no margin
        c = _case(seed, kind, split, margin)
        if c is not None:
            return c
    return None


def unframe_box(b, swap: bool, mirror: bool, X: int) -> list[int]:
    """An inclusive box [x0, x1, y0, y1, z0, z1] from the canonical frame back to the model's."""
    x0, x1, y0, y1, z0, z1 = b
    if mirror:
        x0, x1 = X - 1 - x1, X - 1 - x0
    if swap:
        x0, x1, y0, y1 = y0, y1, x0, x1
    return [x0, x1, y0, y1, z0, z1]


def _case(seed: int, kind: str, split: str, margin: int, use_clear: bool = True) -> dict | None:
    rng = random.Random(f"r6-{kind}-{split}-{seed}")
    body, obj = body_for(rng, split, 0.9 if kind == "hinge" else 0.5)
    prob = hinge_problem(body, rng, margin) if kind == "hinge" else leg_problem(body, rng)
    if prob is None:
        return None
    swap, mirror, X = prob["frame"]
    full = unframe(prob["parts"], swap, mirror, X)
    pos = {i: k for k, i in enumerate(prob["view"])}
    if kind == "hinge" and use_clear:
        # one cut through the sheet (code re-lays what crosses it), then the hinge plates
        # free the panel with a channel on every side but the hinge: the gap behind the hinge line, and one row
        # along each side out to the edge (the strip's own rows stay); then hang it on the hinge plates
        w = prob["where"]
        edge = w["edge"] + 1
        channel = [(w["xg"], w["xg"], w["y0"], w["y1"] - 1), (w["xg"], edge, w["y0"] - 1, w["y0"] - 1),
                   (w["xg"], edge, w["y1"], w["y1"])]
        edits = [{"op": "clear", "box": unframe_box((a, b, c, d, w["z0"], w["z1"]), swap, mirror, X)}
                 for a, b, c, d in channel]
        edits += [{"op": "add", "part": q} for q in unframe([a for a in prob["adds"]
                                                             if a["type"] in ("bar_plate", "clip_plate")],
                                                            swap, mirror, X)]
    else:
        edits = [{"op": "remove", "index": pos[i]} if q is None else
                 {"op": "set", "index": pos[i], "part": unframe([q], swap, mirror, X)[0]}
                 for i, q in prob["sets"].items()]
        edits += [{"op": "add", "part": q} for q in unframe(prob["adds"], swap, mirror, X)]
    reqs = FLAP if kind == "hinge" else LEG
    a = apply_edits(full, edits, local=prob["view"])
    v = check(full, a.parts, a.sources, reqs, stand=False)
    if not v["connected"][0]:
        # small loose bits of the re-laid parts (a corner stud cut off by the gap) are left out, as a builder would
        loose = loose_parts(a.parts)
        mine = {n for n in loose if a.sources[n] is None or n in a.changed}
        if loose and mine == set(loose) and sum(a.parts[n]["w"] * a.parts[n]["l"] for n in loose) <= 6:
            drop = {json.dumps(a.parts[n], sort_keys=True) for n in loose}
            edits = [{"op": "remove", "index": e["index"]} if e["op"] == "set" and
                     json.dumps(e["part"], sort_keys=True) in drop else e for e in edits
                     if not (e["op"] == "add" and json.dumps(e["part"], sort_keys=True) in drop)]
            a = apply_edits(full, edits, local=prob["view"])
            v = check(full, a.parts, a.sources, reqs, stand=False)
    if not all(v[k][0] for k in v if k in KEEP or k.endswith("_surfaces")):
        return None
    level = rng.choice(("L1", "L2", "L3"))
    text = (hinge_text if kind == "hinge" else leg_text)(prob, level, obj)
    return dict(id=f"{kind}_{split}_{seed}", kind=kind, level=level, object=obj, full=full, view=prob["view"],
                instruction=text, requirements=reqs, answer=edits)


def loose_parts(parts: list[dict]) -> list[int]:
    """Parts outside the largest group joined by studs and joints."""
    from .kinematics import analyse

    bodies, joints = analyse(parts)
    link: dict = {}
    for j in joints:
        link.setdefault(j[0], set()).add(j[1])
        link.setdefault(j[1], set()).add(j[0])
    groups, seen = [], set()
    for b0 in bodies:
        if b0 in seen:
            continue
        todo, g = [b0], []
        seen.add(b0)
        while todo:
            b = todo.pop()
            g += bodies[b]
            for c in link.get(b, ()):
                if c not in seen:
                    seen.add(c)
                    todo.append(c)
        groups.append(g)
    groups.sort(key=len)
    return [n for g in groups[:-1] for n in g]


def habitual_mistake(c: dict, rng: random.Random | None = None) -> list[dict]:
    """A realistic wrong answer: with clear, only the gap behind the hinge cut (the panel still held at its sides)
    or a hinge plate one stud off; without clear, adding without removing or shortening anything."""
    clears = [e for e in c["answer"] if e["op"] == "clear"]
    if not clears:
        return [e for e in c["answer"] if e["op"] == "add"]
    rng = rng or random.Random(0)
    if rng.random() < 0.5:
        return clears[:1] + [e for e in c["answer"] if e["op"] == "add"]
    bad = copy.deepcopy(c["answer"])
    adds = [e for e in bad if e["op"] == "add"]
    e = rng.choice(adds)
    e["part"][rng.choice(("x", "y"))] += rng.choice((-1, 1))
    return bad


def row(c: dict, rng: random.Random) -> dict:
    chat = [{"role": "system", "content": RULES},
            {"role": "user", "content": c["instruction"] + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(c['full'][i])}" for k, i in enumerate(c["view"]))}]
    turns = 1
    if c["kind"] == "hinge" and rng.random() < 0.4:
        bad = habitual_mistake(c, rng)
        a = apply_edits(c["full"], bad, local=c["view"])
        v = check(c["full"], a.parts, a.sources, c["requirements"], stand=False)
        keep = {k: x for k, x in v.items() if k in KEEP or k.endswith("_surfaces")}
        issues = explain(c["full"], a.parts, a.sources, a.changed, keep, c["view"])
        if issues:
            chat += [{"role": "assistant", "content": json.dumps(bad)},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
            turns = 2
    chat.append({"role": "assistant", "content": json.dumps(c["answer"])})
    return {"messages": chat, "family": f"primitive_{c['kind']}", "mode": c["level"], "case": c["id"], "turns": turns}


def _job(args):
    seed, kind, split = args
    try:
        return case(seed, kind, split)
    except Exception as e:  # noqa: BLE001 - an odd body is skipped, and counted
        return {"error": f"{type(e).__name__}: {e}"[:200], "seed": seed, "kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--hinge-n", type=int, default=16000, help="hinge seeds tried (most bodies have no thin edge)")
    ap.add_argument("--leg-n", type=int, default=1000, help="leg seeds tried")
    ap.add_argument("--test-n", type=int, default=20, help="held-out cases per primitive (unseen car shells too)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if (args.test / "cases.json").exists():
        raise ValueError("refusing to overwrite a frozen test set")
    jobs = [(s, "hinge", "train") for s in range(args.hinge_n)] + [(s, "leg", "train") for s in range(args.leg_n)]
    tests = [(100000 + s, "hinge", "val") for s in range(args.test_n * 25)] + \
            [(100000 + s, "leg", "val") for s in range(args.test_n * 3)]
    with ProcessPoolExecutor(args.workers) as pool:
        train = list(pool.map(_job, jobs, chunksize=8))
        test = list(pool.map(_job, tests, chunksize=8))
    errors = [c for c in train + test if c and "error" in c]
    train = [c for c in train if c and "error" not in c]
    picked, per = [], {}
    for c in test:
        if c and "error" not in c and per.get(c["kind"], 0) < args.test_n:
            picked.append(c)
            per[c["kind"]] = per.get(c["kind"], 0) + 1
    rng = random.Random(0)
    rows = [row(c, rng) for c in train]
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    args.test.mkdir(parents=True, exist_ok=True)
    (args.test / "cases.json").write_text(json.dumps({"cases": picked}))
    count = lambda k, lv: sum(r["family"] == f"primitive_{k}" and r["mode"] == lv for r in rows)
    print(json.dumps({"train_rows": len(rows), "by": {f"{k}/{lv}": count(k, lv) for k in ("hinge", "leg")
                                                      for lv in ("L1", "L2", "L3")},
                      "two_turn": sum(r["turns"] == 2 for r in rows), "test": per, "errors": len(errors),
                      "error_examples": errors[:3]}, indent=1))


if __name__ == "__main__":
    main()
