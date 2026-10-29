"""BrickGPT's brick format on top of SAMgo's checkers.

A line is `hxw (x,y,z)`: a 1-brick-tall brick filling x..x+h-1, y..y+w-1 at layer z of a 20x20x20 grid
(exactly as in BrickGPT's Brick.slice). Validity per brick (their rejection-sampling rules): well formed,
an allowed size, inside the grid, no collision. Stability uses our force-balance LP with their stud
strength (0.98 N per stud, a 2x4 brick weighing 2.16 g), and rollback cuts to the shortest unstable prefix.
"""

import re
from dataclasses import dataclass

import numpy as np

from samgo.stability import stability

WORLD = 20
ALLOWED = {(2, 4), (4, 2), (2, 6), (6, 2), (1, 2), (2, 1), (1, 4), (4, 1), (1, 6), (6, 1), (1, 8), (8, 1), (1, 1), (2, 2)}
LINE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\)")
# Their stud strength in our units (one plate-sized cell = 1): 0.98 N over the weight of a third of a
# 1x1 brick (2.16 g / 8 cells / 3 plates) is ~1100.
CLUTCH = 0.98 / (2.16e-3 / 8 / 3 * 9.81)
# Part numbers from BrickGPT's brick_library.json (LDraw).
PART_ID = {(1, 1): "3005", (1, 2): "3004", (1, 4): "3010", (1, 6): "3009", (1, 8): "3008",
           (2, 2): "3003", (2, 4): "3001", (2, 6): "2456"}


@dataclass(frozen=True)
class Brick:
    h: int
    w: int
    x: int
    y: int
    z: int

    @property
    def cells(self):
        return slice(self.x, self.x + self.h), slice(self.y, self.y + self.w), self.z

    def txt(self) -> str:
        return f"{self.h}x{self.w} ({self.x},{self.y},{self.z})"

    def piece(self, color: str = "light_gray") -> dict:
        """As a SAMgo piece: z in plates (one brick = 3)."""
        return {"w": self.h, "l": self.w, "type": "brick", "x": self.x, "y": self.y, "z": 3 * self.z, "color": color}

    def ldr(self, color: int = 71) -> str:
        """LDraw line, as BrickGPT's Brick.to_ldr: 20 LDU per stud, 24 per brick, -y up."""
        part = PART_ID[tuple(sorted((self.h, self.w)))]
        matrix = "0 0 1 0 1 0 -1 0 0" if self.h <= self.w else "-1 0 0 0 1 0 0 0 -1"
        x, z, y = (self.x + self.h / 2) * 20, (self.y + self.w / 2) * 20, -self.z * 24
        return f"1 {color} {x:g} {y:g} {z:g} {matrix} {part}.dat"


def parse(line: str) -> Brick | None:
    m = LINE.fullmatch(line.strip())
    return Brick(*map(int, m.groups())) if m else None


class Structure:
    def __init__(self, bricks: list[Brick] = ()):
        self.bricks: list[Brick] = []
        self.occ = np.zeros((WORLD,) * 3, dtype=np.int16)
        for b in bricks:
            self.add(b)

    def reject_reason(self, line: str) -> str | None:
        """None if the line is a brick that can be placed now; else why not (BrickGPT's rejection rules)."""
        b = parse(line)
        if b is None:
            return "malformed"
        if (b.h, b.w) not in ALLOWED:
            return "not_in_library"
        if b.x < 0 or b.y < 0 or b.x + b.h > WORLD or b.y + b.w > WORLD or not 0 <= b.z < WORLD:
            return "out_of_bounds"
        if self.occ[b.cells].any():
            return "collision"
        return None

    def add(self, b: Brick) -> None:
        self.bricks.append(b)
        self.occ[b.cells] += 1

    def truncate(self, n: int) -> None:
        for b in self.bricks[n:]:
            self.occ[b.cells] -= 1
        del self.bricks[n:]

    def pieces(self) -> list[dict]:
        return [b.piece() for b in self.bricks]

    def stable(self, n: int | None = None) -> bool:
        bricks = self.bricks if n is None else self.bricks[:n]
        return bool(bricks) and stability([b.piece() for b in bricks], clutch=CLUTCH).passed

    def first_unstable(self) -> int | None:
        """Index of the brick whose placement first makes the build unstable (None if stable).

        Binary search on prefixes, assuming instability persists once it appears; BrickGPT cuts at the
        first brick whose stability score fails, which is the same point for bottom-up builds.
        """
        n = len(self.bricks)
        if self.stable():
            return None
        lo, hi = 1, n  # stable(lo - 1 bricks) is unknown for lo=1 (empty counts as stable)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.stable(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def txt(self) -> str:
        return "\n".join(b.txt() for b in self.bricks)

    def ldr(self, name: str = "model") -> str:
        lines = [f"0 {name}", f"0 Name: {name}.ldr", "0 Author: SAMgo (Muse Glimmer)"]
        for b in self.bricks:
            lines += [b.ldr(), "0 STEP"]
        return "\n".join(lines) + "\n"


def from_txt(text: str) -> Structure:
    s = Structure()
    for line in text.strip().splitlines():
        b = parse(line)
        if b is not None:
            s.add(b)
    return s
