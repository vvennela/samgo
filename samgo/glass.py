"""Glass windows for cars, found from the shape: the cabin's walls between beltline and roof become glass.

Photo colors are unreliable here (gray cars and tinted windows voxelize to the same grays), so the windows come
from geometry: the cabin is the stack of layers above the beltline, where each layer is clearly shorter than the
body below it. Its outer wall becomes glass, except the corner columns (A/C pillars) and a middle column
(B pillar) on the sides, and the roof layer. Only colors change, never geometry, so every structural and
functional check is unaffected; colors.legalize then splits any glass piece LEGO never made in that size.
"""

GLASS = "trans_black"
BODY_TYPES = ("plate", "brick")
HEIGHT = {"plate": 1, "brick": 3}


def _cells(p: dict):
    for x in range(p["x"], p["x"] + p["w"]):
        for y in range(p["y"], p["y"] + p["l"]):
            for z in range(p["z"], p["z"] + HEIGHT[p["type"]]):
                yield x, y, z


def add_glass(parts: list[dict], shrink: float = 0.8) -> tuple[list[dict], int]:
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

    def glassy(x: int, y: int, z: int) -> bool:
        layer = cells[z]
        xs, ys = [c[0] for c in layer], [c[1] for c in layer]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = y in (y0, y1) or (x, y - 1) not in layer or (x, y + 1) not in layer
        end = x in (x0, x1) or (x - 1, y) not in layer or (x + 1, y) not in layer
        corner = (x in (x0, x1)) and (y in (y0, y1))
        b_pillar = side and not end and x == (x0 + x1) // 2
        return (side or end) and not corner and not b_pillar

    from .colors import NAMED, exists
    from .parts import LDRAW_IDS

    def made_in_glass(p: dict) -> bool:  # only sizes LEGO produced in this color, so no piece is ever split
        pid = LDRAW_IDS.get((p["type"], *sorted((p["w"], p["l"]))))
        return pid is not None and exists(pid, NAMED[GLASS])

    out, n = [], 0
    for p in parts:
        if p["type"] in BODY_TYPES and p["z"] in window_z and made_in_glass(p) \
                and all(z in window_z and glassy(x, y, z) for x, y, z in _cells(p)):
            p = p | {"color": GLASS}
            n += 1
        out.append(p)
    return out, n
