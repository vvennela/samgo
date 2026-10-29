"""Shape-checked decoding: the model still places every brick; the verifier keeps it inside *your* object.

Wraps a decoder structure (BrickGPT's 8 bricks, or the v2 35-part palette) with a target shape from S2
(photo -> SAM 3.1 -> SAM 3D -> voxels) and adds:
- `reject_reason`: a part that fills a cell outside the target (allowing `tolerance` cells at its edges)
  is rejected and resampled, like a collision;
- `allow_end`: the model may only stop once the build covers `min_coverage` of the target;
- `score`: for best-of-k sampling, how many still-empty target cells a candidate part fills;
- `metrics`: coverage, precision and IoU against the target;
- `colors`: each part takes the most common photo color among the target cells it covers.
See decoder.md.
"""

from collections import Counter

import numpy as np
from scipy import ndimage

from samgo.parts import COLORS, COLOR_INDEX

LDRAW_CODE = {i: COLORS[name][0] for name, i in COLOR_INDEX.items()}


def fit_target(target: np.ndarray, grid: tuple[int, int, int], plates_per_unit: int) -> np.ndarray:
    """An S2 target (x, y, plate z; color codes) on the decoder's grid. `plates_per_unit` is 3 for BrickGPT's
    brick-tall layers (a layer is filled where most of its 3 plates are) and 1 for the v2 plate grid."""
    X, Y, Z = target.shape
    layers = -(-Z // plates_per_unit)
    out = np.zeros(grid, dtype=np.int32)
    for k in range(min(layers, grid[2])):
        chunk = target[:, :, k * plates_per_unit:(k + 1) * plates_per_unit]
        filled = (chunk > 0).sum(axis=2) * 2 > chunk.shape[2]
        for x, y in zip(*np.nonzero(filled)):
            if x < grid[0] and y < grid[1]:
                out[x, y, k] = Counter(int(c) for c in chunk[x, y] if c > 0).most_common(1)[0][0]
    return out


def _cells(part) -> tuple[slice, slice, slice]:
    """Grid cells of a parsed part: BrickGPT bricks are one layer tall, v2 parts carry their own box."""
    if hasattr(part, "box"):
        return part.box
    return slice(part.x, part.x + part.h), slice(part.y, part.y + part.w), slice(part.z, part.z + 1)


class ShapeChecked:
    def __init__(self, base, target: np.ndarray, parse_line, tolerance: int = 1, min_coverage: float = 0.9):
        self.base, self.parse, self.min_coverage = base, parse_line, min_coverage
        self.target = target > 0
        self.color = target
        # Allowed region: the target grown by `tolerance` cells, so near misses on thin parts are kept.
        self.allowed = ndimage.binary_dilation(self.target, iterations=tolerance) if tolerance else self.target
        self.built = np.zeros(target.shape, dtype=bool)
        self.n_target = int(self.target.sum())

    # The decoder's structure interface, delegated.
    @property
    def bricks(self):
        return self.base.bricks

    def stable(self, n=None):
        return self.base.stable(n)

    def first_unstable(self):
        return self.base.first_unstable()

    def txt(self) -> str:
        return self.base.txt()

    def add(self, part) -> None:
        self.base.add(part)
        self.built[_cells(part)] = True

    def truncate(self, n: int) -> None:
        self.base.truncate(n)
        self.built[:] = False
        for p in self.base.bricks:
            self.built[_cells(p)] = True

    def reject_reason(self, line: str) -> str | None:
        why = self.base.reject_reason(line)
        if why is not None:
            return why
        if not self.allowed[_cells(self.parse(line))].all():
            return "outside_shape"
        return None

    def coverage(self) -> float:
        return float((self.built & self.target).sum()) / max(self.n_target, 1)

    def allow_end(self) -> bool:
        return self.coverage() >= self.min_coverage

    def score(self, line: str) -> float:
        """Still-empty target cells this part would fill (higher is better); -1 if it may not be placed."""
        if self.reject_reason(line) is not None:
            return -1.0
        c = _cells(self.parse(line))
        return float((self.target[c] & ~self.built[c]).sum())

    def metrics(self) -> dict:
        inter = int((self.built & self.target).sum())
        built = int(self.built.sum())
        union = int((self.built | self.target).sum())
        return {"coverage": round(inter / max(self.n_target, 1), 3), "precision": round(inter / max(built, 1), 3),
                "iou": round(inter / max(union, 1), 3), "target_cells": self.n_target, "built_cells": built}

    def colors(self) -> list[int]:
        """LDraw color code per part: the most common photo color among the target cells it covers."""
        out = []
        for p in self.base.bricks:
            codes = self.color[_cells(p)]
            codes = codes[codes > 0]
            out.append(LDRAW_CODE[Counter(codes.tolist()).most_common(1)[0][0]] if codes.size else 71)
        return out
