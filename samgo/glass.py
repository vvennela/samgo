"""Glass windows for cars, found from the shape: the cabin's walls between beltline and roof become glass.

Photo colors are unreliable here (gray cars and tinted windows voxelize to the same grays), so the windows come
from geometry: the cabin is the stack of layers above the beltline, where each layer is clearly shorter than the
body below it. Its outer wall becomes glass, except the corner columns (A/C pillars) and a middle column
(B pillar) on the sides, and the roof layer. Only colors change, never geometry, so every structural and
functional check is unaffected; colors.legalize then splits any glass piece LEGO never made in that size.
"""

GLASS = "trans_clear"  # see-through; LEGO makes far more sizes in trans-clear than trans-black
BODY_TYPES = ("plate", "brick")
HEIGHT = {"plate": 1, "brick": 3}


def _cells(p: dict):
    for x in range(p["x"], p["x"] + p["w"]):
        for y in range(p["y"], p["y"] + p["l"]):
            for z in range(p["z"], p["z"] + HEIGHT[p["type"]]):
                yield x, y, z


def add_glass(parts: list[dict], shrink: float = 0.8, cut: bool = True) -> tuple[list[dict], int]:
    """(parts with the cabin's windows made glass, how many parts became glass)."""
    body = [p for p in parts if p["type"] in BODY_TYPES]
    if not body:
        return parts, 0
    cells: dict[int, set] = {}
    for p in body:
        for x, y, z in _cells(p):
            cells.setdefault(z, set()).add((x, y))
    levels = sorted(cells)
    length = {z: max(x for x, _ in cells[z]) - min(x for x, _ in cells[z]) + 1 for z in levels}
    lower = max(length.values())
    cabin = [z for z in levels if length[z] <= shrink * lower]
    if not cabin:
        return parts, 0
    belt, roof = min(cabin), max(levels)
    window_z = set(range(belt, roof))  # the roof layer stays solid

    def near_cabin(x: int, z: int) -> bool:
        """Within two studs of the layer above (a windshield or rear-window step), so not the trunk or hood."""
        above = [a for a, _ in cells.get(z + 1, ())]
        return bool(above) and min(above) - 2 <= x <= max(above) + 2

    def glassy(x: int, y: int, z: int) -> bool:
        layer = cells[z]
        xs, ys = [c[0] for c in layer], [c[1] for c in layer]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = y in (y0, y1) or (x, y - 1) not in layer or (x, y + 1) not in layer
        end = x in (x0, x1) or (x - 1, y) not in layer or (x + 1, y) not in layer
        corner = (x in (x0, x1)) and (y in (y0, y1))
        b_pillar = side and not end and x == (x0 + x1) // 2
        if end and not side and not near_cabin(x, z):
            return False  # the car's front or rear face, not a windshield step
        if side and not end:  # a side window only under the cabin: every layer above, up to the roof, covers x
            for z2 in range(z + 1, roof + 1):
                above = [a for a, _ in cells.get(z2, ())]
                if not above or not min(above) <= x <= max(above):
                    return False
        return (side or end) and not corner and not b_pillar

    from .colors import NAMED, exists
    from .parts import LDRAW_IDS

    def made_in_glass(p: dict) -> bool:  # only sizes LEGO produced in this color, so no piece is ever split
        pid = LDRAW_IDS.get((p["type"], *sorted((p["w"], p["l"]))))
        return pid is not None and exists(pid, NAMED[GLASS])

    occupied = {c for p in body for c in _cells(p)}

    # the air around the model, flood-filled from outside its box: faces into the hollow cabin do not show
    xs_, ys_, zs_ = zip(*occupied)
    box = (min(xs_) - 1, max(xs_) + 1, min(ys_) - 1, max(ys_) + 1, min(zs_) - 1, max(zs_) + 1)
    outside, todo = set(), [(box[0], box[2], box[4])]
    while todo:
        c = todo.pop()
        if c in outside or c in occupied:
            continue
        x, y, z = c
        if not (box[0] <= x <= box[1] and box[2] <= y <= box[3] and box[4] <= z <= box[5]):
            continue
        outside.add(c)
        todo += [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]

    top_of = {}
    for (x, y, z) in occupied:
        top_of[(x, y)] = max(top_of.get((x, y), z), z)

    def sky(x: int, y: int, z: int) -> bool:  # nothing anywhere above: a windshield step, not the cabin floor
        return top_of[(x, y)] == z

    def visible(c) -> bool:
        x, y, z = c
        # seen from the sides, the ends or above; a face looking down (the cabin's ceiling) shows to no one
        return sky(x, y, z) or any(n in outside for n in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z),
                                                          (x, y - 1, z)))

    roof_xs = [a for a, _ in cells[roof]]

    def greenhouse(x: int, z: int) -> bool:
        """Under the roof, widening by half a stud per plate going down: the windshield's and rear window's
        slopes, not the hood or the trunk."""
        reach = 0.5 * (roof - z) + 0.5
        return min(roof_xs) - reach <= x <= max(roof_xs) + reach

    def window(c) -> bool:
        x, y, z = c
        if z not in window_z or not greenhouse(x, z):
            return False
        # the cabin's outline, or the top of a stepped windshield / rear window (open above), minus the pillars
        return glassy(x, y, z) or (sky(x, y, z) and near_cabin(x, z) and not _pillar(x, y, z))

    def _pillar(x: int, y: int, z: int) -> bool:
        layer = cells[z]
        ys = [c[1] for c in layer]
        return y in (min(ys), max(ys))  # the windshield's sides are the A pillars

    # Windows are cells, decided before any part is recolored: the visible cells of the cabin's outline and of its
    # stepped windshield / rear window. A mirror-symmetric body gets mirror-symmetric windows.
    win = {c for c in occupied if visible(c) and window(c)}
    return glaze(parts, win, cut=cut)


def visible_cells(parts: list[dict]) -> set:
    """Cells of the body seen from outside: from the sides, the ends or above (flood-filled air from outside the
    model's box, so faces into a hollow do not count)."""
    body = [p for p in parts if p.get("type") in BODY_TYPES]
    occupied = {c for p in body for c in _cells(p)}
    if not occupied:
        return set()
    xs_, ys_, zs_ = zip(*occupied)
    box = (min(xs_) - 1, max(xs_) + 1, min(ys_) - 1, max(ys_) + 1, min(zs_) - 1, max(zs_) + 1)
    outside, todo = set(), [(box[0], box[2], box[4])]
    while todo:
        c = todo.pop()
        if c in outside or c in occupied:
            continue
        x, y, z = c
        if not (box[0] <= x <= box[1] and box[2] <= y <= box[3] and box[4] <= z <= box[5]):
            continue
        outside.add(c)
        todo += [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]
    return {(x, y, z) for (x, y, z) in occupied
            if any(n in outside for n in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1)))}


def glaze(parts: list[dict], win: set, cut: bool = True) -> tuple[list[dict], int]:
    """Make the cells `win` glass, for any object: a part wholly in them becomes glass; a plate across their edge is
    cut into glass and body pieces, kept only if the build holds together as well as before; glass only in sizes
    LEGO made in trans-clear. A mirror-symmetric body keeps symmetric glass. (parts, how many parts are glass)."""
    from .colors import NAMED, exists
    from .parts import LDRAW_IDS

    body = [p for p in parts if p.get("type") in BODY_TYPES]
    if not body or not win:
        return parts, 0

    def made_in_glass(p: dict) -> bool:
        pid = LDRAW_IDS.get((p["type"], *sorted((p["w"], p["l"]))))
        return pid is not None and exists(pid, NAMED[GLASS])

    occupied = {c for p in body for c in _cells(p)}
    window_z = {z for _, _, z in win}
    y_lo, y_hi = min(y for _, y, _ in occupied), max(y for _, y, _ in occupied)
    mirror = lambda c: (c[0], y_lo + y_hi - c[1], c[2])
    symmetric = sum(mirror(c) in occupied for c in occupied) >= 0.95 * len(occupied)
    if symmetric:
        win = {c for c in win if mirror(c) in win}

    from .primitives import tile_cells
    from .primitives import loose_parts

    def issues(ps):  # parts not attached to the rest (through studs and joints): a cut must not add any
        return len(loose_parts(ps))

    # A part wholly in a window becomes glass; a part across a window's edge (a long plate running from the
    # windshield into a pillar or the roof) is cut at the edge into glass and body pieces, kept only if the build
    # holds together as well as before. Glass pieces only in sizes LEGO made in trans-clear.
    out, base = list(parts), issues(parts)
    for i, p in enumerate(parts):
        if p["type"] not in BODY_TYPES or p["z"] not in window_z:
            continue
        cs = list(_cells(p))
        glass_cols = {(x, y) for x, y, z in cs if (x, y, z) in win}
        if not glass_cols:
            continue
        if len(glass_cols) == p["w"] * p["l"] and made_in_glass(p):
            out[i] = p | {"color": GLASS}
            continue
        if p["type"] != "plate" or not cut:
            continue  # bricks are not cut (a cut brick changes how the layers above grip it); cut=False: whole parts
        rest = {(x, y) for x, y, _ in cs} - glass_cols
        fewest = lambda cs: min((tile_cells(cs, 0), tile_cells(cs, 1)), key=len)
        pieces = [dict(p, x=x, y=y, w=w, l=l, color=GLASS) for x, y, w, l in fewest(glass_cols)]
        pieces = [q if made_in_glass(q) else q | {"color": p["color"]} for q in pieces]
        if not any(q["color"] == GLASS for q in pieces):
            continue
        pieces += [dict(p, x=x, y=y, w=w, l=l) for x, y, w, l in fewest(rest)]
        trial = out[:i] + [pieces[0]] + out[i + 1:] + pieces[1:]
        if issues(trial) <= base:
            out = trial
    if symmetric:  # pieces that could be glass on one side only go back to body color
        while True:
            glass = {c for q in out if q.get("color") == GLASS and q["type"] in BODY_TYPES for c in _cells(q)}
            lone = [k for k, q in enumerate(out) if q.get("color") == GLASS and q["type"] in BODY_TYPES
                    and not all(mirror(c) in glass for c in _cells(q))]
            if not lone:
                break
            body_color = max((q.get("color") for q in body), key=[q.get("color") for q in body].count)
            for k in lone:
                out[k] = out[k] | {"color": body_color}
    return out, sum(q.get("color") == GLASS for q in out)
