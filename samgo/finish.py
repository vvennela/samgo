"""Finishing pass: swap plain bricks and plates for the 35-part palette's richer parts, so a build reads as LEGO.

Uses brickgpt/v2.py's rule, generalized to any grid and made deterministic and mirror-consistent:
a part is only swapped where that cannot break a connection (it keeps every grip underneath and every stud in
use above):
- exposed plate tops -> tiles;
- bricks with an open top and an open side -> slopes facing that side (curved slopes where a plate fits under);
- 1x2 bricks over an overhang -> inverted slopes;
- 1x4 / 1x6 bricks bridging a gap -> arches;
- 1x1 / 2x2 posts -> round bricks and plates.
Technic beams, axles and wheels are never touched. For a symmetric build, a part and its mirror twin get the
mirrored choice (a slope facing +y becomes one facing -y).
"""

import numpy as np

from .palette import KINDS

HEIGHT = {k: v.height for k, v in KINDS.items()} | {"technic_brick": 3}
FIXED = {"technic_brick", "axle", "wheel"}
MIRROR_FACING = {"+y": "-y", "-y": "+y", "+x": "+x", "-x": "-x"}


class _Grid:
    def __init__(self, parts: list[dict], shape: tuple[int, int, int]):
        self.shape = shape
        self.owner = np.full(shape, -1, dtype=np.int32)
        for i, p in enumerate(parts):
            if p.get("type") in HEIGHT:
                self.owner[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["l"], p["z"]:p["z"] + HEIGHT[p["type"]]] = i

    def empty(self, x0, x1, y0, y1, z0, z1) -> bool:
        X, Y, Z = self.shape
        x0, y0, z0, x1, y1, z1 = max(x0, 0), max(y0, 0), max(z0, 0), min(x1, X), min(y1, Y), min(z1, Z)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return True  # outside the grid is open air
        return bool((self.owner[x0:x1, y0:y1, z0:z1] < 0).all())

    def above_empty(self, p) -> bool:
        top = p["z"] + HEIGHT[p["type"]]
        return self.empty(p["x"], p["x"] + p["w"], p["y"], p["y"] + p["l"], top, top + 1)

    def side_empty(self, p, f) -> bool:
        x, y, w, l, z, h = p["x"], p["y"], p["w"], p["l"], p["z"], HEIGHT[p["type"]]
        if f == "+x":
            return self.empty(x + w, x + w + 1, y, y + l, z, z + h)
        if f == "-x":
            return self.empty(x - 1, x, y, y + l, z, z + h)
        if f == "+y":
            return self.empty(x, x + w, y + l, y + l + 1, z, z + h)
        return self.empty(x, x + w, y - 1, y, z, z + h)

    def below(self, x, y, z) -> bool:
        return z > 0 and self.owner[x, y, z - 1] >= 0


def _slope_facings(p) -> list[str]:
    out = []
    if p["l"] == 2 and (p["w"], 2) in KINDS["slope"].sizes:
        out += ["+y", "-y"]
    if p["w"] == 2 and (p["l"], 2) in KINDS["slope"].sizes:
        out += ["+x", "-x"]
    return out


def _choose(p: dict, grid: _Grid, Y: int, symmetric: bool) -> list[dict]:
    """The finished version of one part (deterministic). Facings prefer the side that is open."""
    base = {k: v for k, v in p.items() if k not in ("type", "facing")}
    open_top = grid.above_empty(p)
    size = tuple(sorted((p["w"], p["l"])))
    if p["type"] == "plate" and open_top and size in KINDS["tile"].sizes:
        return [base | {"type": "tile"}]
    if p["type"] == "brick":
        facings = [f for f in _slope_facings(p) if grid.side_empty(p, f)]
        if symmetric and p["y"] + p["l"] / 2 == Y / 2:
            facings = [f for f in facings if f[1] == "x"]  # a part on the mirror line may only slope fore or aft
        if open_top and facings:
            # The side facing out of the body: for a symmetric build, the half the part is in decides +y / -y.
            centre_y = p["y"] + p["l"] / 2
            prefer = ("+y" if centre_y > Y / 2 else "-y") if symmetric else facings[0]
            f = prefer if prefer in facings else facings[0]
            return [base | {"type": "slope", "facing": f}]
        if open_top and size in KINDS["tile"].sizes:
            return [base | {"type": "plate"}, base | {"type": "plate", "z": p["z"] + 1},
                    base | {"type": "tile", "z": p["z"] + 2}]
        if p["z"] > 0 and size == (1, 2):
            cells = [(x, y) for x in range(p["x"], p["x"] + p["w"]) for y in range(p["y"], p["y"] + p["l"])]
            held = [grid.below(x, y, p["z"]) for x, y in cells]
            if sum(held) == 1:
                hx, hy = cells[held.index(True)]
                ox, oy = cells[held.index(False)]
                f = ("+x" if ox > hx else "-x") if ox != hx else ("+y" if oy > hy else "-y")
                return [base | {"type": "invslope", "facing": f}]
        if p["z"] > 0 and size in ((1, 4), (1, 6)):
            cells = ([(x, p["y"]) for x in range(p["x"], p["x"] + p["w"])] if p["w"] > p["l"]
                     else [(p["x"], y) for y in range(p["y"], p["y"] + p["l"])])
            held = [grid.below(x, y, p["z"]) for x, y in cells]
            if held[0] and held[-1] and not any(held[1:-1]):
                return [base | {"type": "arch"}]
    if p["type"] in ("brick", "plate") and (p["w"], p["l"]) in ((1, 1), (2, 2)):
        if sum(grid.side_empty(p, f) for f in ("+x", "-x", "+y", "-y")) >= 3:
            return [base | {"type": "roundbrick" if p["type"] == "brick" else "roundplate"}]
    return [p]


def finish(parts: list[dict], shape: tuple[int, int, int]) -> list[dict]:
    """The build with richer parts swapped in where they cannot break a connection; running gear untouched."""
    grid = _Grid(parts, shape)
    Y = shape[1]
    solids = [p for p in parts if p.get("type") in ("brick", "plate")]
    keys = {(p["type"], p["w"], p["l"], p["x"], p["y"], p["z"]) for p in solids}
    symmetric = all((p["type"], p["w"], p["l"], p["x"], Y - p["y"] - p["l"], p["z"]) in keys for p in solids)
    out = []
    for p in parts:
        out += [p] if p.get("type") in FIXED or p.get("type") not in ("brick", "plate") else _choose(p, grid, Y, symmetric)
    return out
