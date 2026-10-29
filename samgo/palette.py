"""The SAMgo v2 palette: 31 parts of 9 kinds (bricks, plates, tiles, 45-degree and inverted slopes, curved
slopes, round bricks and plates, arches), with what each part can connect to.

Grid: x, y in studs, z in plates. A placed part has a footprint w (along x) by l (along y), a height in
plates, and for sloped parts a facing: the direction its slope goes down ("+x", "-x", "+y", "-y"),
always along the footprint's 2-stud side. Two parts connect where a stud on top of the lower one meets a
gripping cell underneath the upper one. Slopes have studs only on their high row; tiles and curved
slopes have none; arches grip only at their two ends; inverted slopes grip only under their high row.

Model line format:  `<kind> <w>x<l> (x,y,z)` plus ` <facing>` for slopes, e.g. `slope 1x2 (4,6,9) -y`.
LDraw ids were checked against the official library (Oct 29); placement follows each part file's origin.
"""

import re
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

FACINGS = ("+x", "-x", "+y", "-y")


@dataclass(frozen=True)
class Kind:
    height: int              # plates
    sizes: dict              # (a, b) with a <= b -> LDraw id; a is the width across, b the length/depth
    sloped: bool = False     # has a facing; depth b must be 2
    studs: str = "all"       # top studs: all | high | none
    grips: str = "all"       # bottom: all | high | ends
    origin: str = "top"      # LDraw origin: top centre | back (top centre of high row) | bottom centre


KINDS = {
    "brick": Kind(3, {(1, 1): "3005", (1, 2): "3004", (1, 3): "3622", (1, 4): "3010", (1, 6): "3009",
                      (2, 2): "3003", (2, 3): "3002", (2, 4): "3001"}),
    "plate": Kind(1, {(1, 1): "3024", (1, 2): "3023b", (1, 4): "3710", (1, 6): "3666", (2, 2): "3022",
                      (2, 4): "3020", (2, 6): "3795", (4, 4): "3031"}),
    "tile": Kind(1, {(1, 2): "3069b", (1, 4): "2431", (2, 2): "3068b"}, studs="none"),
    "slope": Kind(3, {(1, 2): "3040b", (2, 2): "3039", (4, 2): "3037"}, sloped=True, studs="high", origin="back"),
    "invslope": Kind(3, {(1, 2): "3665a"}, sloped=True, grips="high", origin="back"),
    "curve": Kind(2, {(1, 2): "11477", (2, 2): "15068"}, sloped=True, studs="none", origin="bottom"),
    "roundbrick": Kind(3, {(1, 1): "3062b", (2, 2): "3941"}),
    "roundplate": Kind(1, {(1, 1): "6141", (2, 2): "4032a"}),
    "arch": Kind(3, {(1, 4): "3659", (1, 6): "3455"}, grips="ends"),
}
# Sloped kinds are listed as (width across, depth along the facing); the depth is always 2.
LINE = re.compile(r"([a-z]+) (\d+)x(\d+) \((-?\d+),(-?\d+),(-?\d+)\)(?: ([+-][xy]))?")


def count() -> int:
    return sum(len(k.sizes) for k in KINDS.values())


class PaletteError(ValueError):
    pass


@dataclass(frozen=True)
class Part:
    kind: str
    w: int
    l: int
    x: int
    y: int
    z: int
    facing: str | None = None
    color: str = "light_gray"

    @property
    def spec(self) -> Kind:
        return KINDS[self.kind]

    @property
    def h(self) -> int:
        return self.spec.height

    @property
    def box(self) -> tuple[slice, slice, slice]:
        return slice(self.x, self.x + self.w), slice(self.y, self.y + self.l), slice(self.z, self.z + self.h)

    def _rows(self, which: str) -> set[tuple[int, int]]:
        """Cells (x, y) of the whole footprint, the high row, or the two ends."""
        cells = {(x, y) for x in range(self.x, self.x + self.w) for y in range(self.y, self.y + self.l)}
        if which == "all":
            return cells
        if which == "none":
            return set()
        if which == "high":  # the row opposite the facing
            axis, sign = self.facing[1], self.facing[0]
            if axis == "x":
                keep = self.x if sign == "+" else self.x + self.w - 1
                return {c for c in cells if c[0] == keep}
            keep = self.y if sign == "+" else self.y + self.l - 1
            return {c for c in cells if c[1] == keep}
        if which == "ends":  # the two end cells along the long side
            if self.w >= self.l:
                return {c for c in cells if c[0] in (self.x, self.x + self.w - 1)}
            return {c for c in cells if c[1] in (self.y, self.y + self.l - 1)}
        raise ValueError(which)

    @property
    def studs(self) -> set[tuple[int, int]]:
        return self._rows(self.spec.studs)

    @property
    def grips(self) -> set[tuple[int, int]]:
        return self._rows(self.spec.grips)

    def line(self) -> str:
        return f"{self.kind} {self.w}x{self.l} ({self.x},{self.y},{self.z})" + (f" {self.facing}" if self.facing else "")

    def ldraw_id(self) -> str:
        spec = self.spec
        if spec.sloped:
            across = self.w if self.facing[1] == "y" else self.l
            return spec.sizes[(across, 2)]
        return spec.sizes[tuple(sorted((self.w, self.l)))]

    def ldr(self, color: int = 71) -> str:
        """One LDraw line. LDraw X = our x, LDraw Z = our y, LDraw -Y = up; 20 LDU per stud, 8 per plate."""
        spec = self.spec
        cx, cy = (self.x + self.w / 2) * 20, (self.y + self.l / 2) * 20
        top, bottom = -(self.z + self.h) * 8, -self.z * 8
        if not spec.sloped:
            # Parts are modelled long along LDraw X; one long along our y is turned a quarter.
            matrix = "1 0 0 0 1 0 0 0 1" if self.w >= self.l else "0 0 1 0 1 0 -1 0 0"
            return f"1 {color} {cx:g} {top:g} {cy:g} {matrix} {self.ldraw_id()}.dat"
        # Sloped parts are modelled facing LDraw -Z (our -y); turn them to face this part's facing.
        matrix = {"-y": "1 0 0 0 1 0 0 0 1", "+y": "-1 0 0 0 1 0 0 0 -1",
                  "+x": "0 0 -1 0 1 0 1 0 0", "-x": "0 0 1 0 1 0 -1 0 0"}[self.facing]
        if spec.origin == "bottom":
            return f"1 {color} {cx:g} {bottom:g} {cy:g} {matrix} {self.ldraw_id()}.dat"
        # Origin at the top centre of the high row.
        hx, hy = cx, cy
        if self.facing == "-y":
            hy = (self.y + self.l - 0.5) * 20
        elif self.facing == "+y":
            hy = (self.y + 0.5) * 20
        elif self.facing == "-x":
            hx = (self.x + self.w - 0.5) * 20
        else:
            hx = (self.x + 0.5) * 20
        return f"1 {color} {hx:g} {top:g} {hy:g} {matrix} {self.ldraw_id()}.dat"


def parse(line: str) -> Part:
    m = LINE.fullmatch(line.strip())
    if not m:
        raise PaletteError("malformed")
    kind, w, l, x, y, z, facing = m.group(1), *map(int, m.group(2, 3, 4, 5, 6)), m.group(7)
    if kind not in KINDS:
        raise PaletteError(f"unknown part {kind!r}")
    spec = KINDS[kind]
    if spec.sloped:
        if facing is None:
            raise PaletteError(f"{kind} needs a facing (+x, -x, +y, -y)")
        depth, across = (l, w) if facing[1] == "y" else (w, l)
        if depth != 2 or (across, 2) not in spec.sizes:
            raise PaletteError(f"no {w}x{l} {kind} facing {facing}")
    else:
        if facing is not None:
            raise PaletteError(f"{kind} has no facing")
        if tuple(sorted((w, l))) not in spec.sizes:
            raise PaletteError(f"no {w}x{l} {kind}")
    return Part(kind, w, l, x, y, z, facing)


def check(parts: list[Part], shape: tuple[int, int, int]) -> dict:
    """Structure check: inside the grid, no overlaps, one connected structure resting on z = 0.

    Returns {"ok", "issues": [(code, [indices])], "components"}.
    """
    X, Y, Z = shape
    issues = []
    out = [i for i, p in enumerate(parts) if p.x < 0 or p.y < 0 or p.z < 0 or p.x + p.w > X or p.y + p.l > Y
           or p.z + p.h > Z]
    if out:
        return {"ok": False, "issues": [("out_of_bounds", out)], "components": None}
    count_ = np.zeros(shape, dtype=np.int16)
    for p in parts:
        count_[p.box] += 1
    overlap = [i for i, p in enumerate(parts) if count_[p.box].max() > 1]
    if overlap:
        issues.append(("overlap", overlap))
    groups = components(parts)
    grounded = [g for g in groups if any(parts[i].z == 0 for i in g)]
    floating = sorted(i for g in groups if g not in grounded for i in g)
    if floating:
        issues.append(("floating", floating))
    if len(grounded) > 1:
        main = max(grounded, key=len)
        issues.append(("disconnected", sorted(i for g in grounded if g is not main for i in g)))
    return {"ok": not issues, "issues": issues, "components": len(groups)}


def contacts(parts: list[Part]) -> list[tuple[int, int, int, int]]:
    """(upper, lower, x, y) for every stud cell where a lower part's stud meets an upper part's grip."""
    tops: dict[tuple[int, int, int], int] = {}
    for i, p in enumerate(parts):
        for x, y in p.studs:
            tops[(x, y, p.z + p.h)] = i
    out = []
    for i, p in enumerate(parts):
        for x, y in p.grips:
            j = tops.get((x, y, p.z))
            if j is not None and j != i:
                out.append((i, j, x, y))
    return out


def components(parts: list[Part]) -> list[list[int]]:
    parent = list(range(len(parts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _, _ in contacts(parts):
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(parts)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


RIM = 0.3  # stud radius in studs
CLUTCH = 1110.0  # per stud, in plate-cell weights (BrickGPT's 0.98 N; see brickgpt/structure.py)


def stability(parts: list[Part], clutch: float = CLUTCH) -> dict:
    """Force and torque balance per part (as samgo.stability), using each part's own studs and grips.

    Returns {"ok", "peak_pull_per_stud" (None if it cannot balance)}.
    """
    if not parts:
        return {"ok": False, "peak_pull_per_stud": None}
    pts = []  # (upper, lower or None, x, y, is_support)
    for i, j, x, y in contacts(parts):
        pts += [(i, j, x + 0.5 + dx, y + 0.5 + dy, False) for dx in (-RIM, RIM) for dy in (-RIM, RIM)]
    for i, p in enumerate(parts):
        if p.z == 0:
            for x, y in p.grips:
                pts += [(i, None, x + cx, y + cy, True) for cx in (0, 1) for cy in (0, 1)]
    n, m = len(pts), len(parts)
    tau = n
    a_eq, b_eq = lil_matrix((3 * m, n + 1)), np.zeros(3 * m)
    centers = [(p.x + p.w / 2, p.y + p.l / 2) for p in parts]
    for i, p in enumerate(parts):
        b_eq[3 * i] = p.w * p.l * p.h * (0.5 if p.spec.sloped and p.kind != "invslope" else 1.0)
    for k, (up, down, x, y, _) in enumerate(pts):
        for piece, sign in ((up, 1.0), (down, -1.0)):
            if piece is None:
                continue
            cx, cy = centers[piece]
            a_eq[3 * piece, k] += sign
            a_eq[3 * piece + 1, k] += sign * (x - cx)
            a_eq[3 * piece + 2, k] += sign * (y - cy)
    a_ub = lil_matrix((n, n + 1))
    for k, p in enumerate(pts):
        if not p[4]:
            a_ub[k, k], a_ub[k, tau] = -1.0, -1.0
    bounds = [(0, None) if p[4] else (None, None) for p in pts] + [(0, None)]
    cost = np.zeros(n + 1)
    cost[tau] = 1.0
    res = linprog(cost, A_ub=a_ub.tocsr(), b_ub=np.zeros(n), A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status != 0:
        return {"ok": False, "peak_pull_per_stud": None}
    peak = 4 * float(res.x[tau])
    return {"ok": peak <= clutch, "peak_pull_per_stud": round(peak, 1)}


def ldr(parts: list[Part], name: str = "model") -> str:
    lines = [f"0 {name}", f"0 Name: {name}.ldr", "0 Author: SAMgo (Muse Glimmer)"]
    for p in sorted(parts, key=lambda p: p.z):
        lines += [p.ldr(), "0 STEP"]
    return "\n".join(lines) + "\n"
