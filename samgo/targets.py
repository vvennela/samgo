"""Procedural targets: seeded shape families on the verifier's grid.

Each target is an (X, Y, Z) int grid of COLOR_INDEX values (0 = empty), with
x and y in studs and z in plates, one connected solid resting on z = 0. These
feed the S4 pass-rate eval and the training curriculum; photo targets come
from S2 instead.
"""

import numpy as np

from .parts import COLOR_INDEX

PALETTE = np.array(list(COLOR_INDEX.values()))


def _grid(x, y, z):
    return np.zeros((x, y, z), dtype=np.int32)


def box(rng):
    g = _grid(*rng.integers([2, 2, 3], [9, 9, 16]))
    g[:] = rng.choice(PALETTE)
    return g


def pyramid(rng):
    base, step = int(rng.integers(6, 13)), int(rng.choice([1, 2, 3]))
    levels = base // 2
    g = _grid(base, base, levels * step)
    for k in range(levels):
        g[k:base - k, k:base - k, k * step:(k + 1) * step] = rng.choice(PALETTE)
    return g


def stairs(rng):
    steps, depth, rise, width = (int(v) for v in rng.integers([3, 1, 2, 2], [7, 3, 7, 7]))
    g = _grid(steps * depth, width, steps * rise)
    for k in range(steps):
        g[k * depth:, :, k * rise:(k + 1) * rise] = rng.choice(PALETTE)
    return g


def table(rng):
    # Tops are at least 2 plates: a 1-plate slab wider than 8 studs can't be tied
    # together by any single part, so it is unbuildable, not hard.
    x, y, legs_h, top_h = (int(v) for v in rng.integers([5, 4, 6, 2], [13, 11, 16, 5]))
    g = _grid(x, y, legs_h + top_h)
    leg = rng.choice(PALETTE)
    for lx in (0, x - 1):
        for ly in (0, y - 1):
            g[lx, ly, :legs_h] = leg
    g[:, :, legs_h:] = rng.choice(PALETTE)
    return g


def chair(rng):
    x, legs_h, seat_h, back_h = (int(v) for v in rng.integers([4, 6, 2, 6], [9, 13, 4, 16]))
    g = _grid(x, x, legs_h + seat_h + back_h)
    wood = rng.choice(PALETTE)
    for lx in (0, x - 1):
        for ly in (0, x - 1):
            g[lx, ly, :legs_h] = wood
    g[:, :, legs_h:legs_h + seat_h] = rng.choice(PALETTE)
    g[:, x - 1, legs_h + seat_h:] = wood
    return g


def arch(rng):
    span, pillar, width, height, lintel = (int(v) for v in rng.integers([2, 1, 1, 6, 1], [9, 3, 4, 19, 4]))
    g = _grid(span + 2 * pillar, width, height + lintel)
    color = rng.choice(PALETTE)
    g[:pillar, :, :height] = color
    g[-pillar:, :, :height] = color
    g[:, :, height:] = rng.choice(PALETTE)
    return g


def mushroom(rng):
    # A thin stem with a wide cap: overhang on every side.
    stem, cap, stem_h, cap_h = (int(v) for v in rng.integers([1, 5, 4, 2], [3, 11, 13, 7]))
    g = _grid(cap, cap, stem_h + cap_h)
    lo = (cap - stem) // 2
    g[lo:lo + stem, lo:lo + stem, :stem_h] = rng.choice(PALETTE)
    g[:, :, stem_h:] = rng.choice(PALETTE)
    return g


def cup(rng):
    x, y, h = (int(v) for v in rng.integers([4, 4, 4], [11, 11, 16]))
    g = _grid(x, y, h)
    g[:] = rng.choice(PALETTE)
    g[1:-1, 1:-1, 1:] = 0
    return g


def ellipsoid(rng):
    rx, ry, rz = rng.uniform([2.5, 2.5, 4], [6.5, 6.5, 14])
    nx, ny, nz = int(2 * rx), int(2 * ry), int(2 * rz)
    x, y, z = np.meshgrid(np.arange(nx) + 0.5 - rx, np.arange(ny) + 0.5 - ry, np.arange(nz) + 0.5 - rz, indexing="ij")
    inside = (x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2 <= 1
    g = np.where(inside, rng.choice(PALETTE), 0).astype(np.int32)
    return _trim(g)


def cylinder(rng):
    r, h = rng.uniform(2, 6.5), int(rng.integers(3, 19))
    n = int(2 * r)
    x, y = np.meshgrid(np.arange(n) + 0.5 - r, np.arange(n) + 0.5 - r, indexing="ij")
    disc = x ** 2 + y ** 2 <= r * r
    g = _grid(n, n, h)
    g[disc] = rng.choice(PALETTE)
    return _trim(g)


def blob(rng):
    # Random overlapping boxes, each touching the ones before it.
    g = _grid(14, 14, 30)
    for k in range(int(rng.integers(2, 6))):
        w, l, h = (int(v) for v in rng.integers([2, 2, 2], [7, 7, 10]))
        if k == 0:
            x, y, z = int(rng.integers(0, 14 - w + 1)), int(rng.integers(0, 14 - l + 1)), 0
        else:
            cx, cy, cz = (int(v) for v in rng.choice(np.argwhere(g > 0)))
            x, y = int(np.clip(cx - rng.integers(0, w), 0, 14 - w)), int(np.clip(cy - rng.integers(0, l), 0, 14 - l))
            z = int(np.clip(cz - rng.integers(0, h), 0, 30 - h))
        g[x:x + w, y:y + l, z:z + h] = rng.choice(PALETTE)
    return _trim(g)


FAMILIES = {f.__name__: f for f in (box, pyramid, stairs, table, chair, arch, mushroom, cup, ellipsoid, cylinder, blob)}


def _trim(g: np.ndarray) -> np.ndarray:
    """Crop to the occupied bounding box and drop anything below the first filled layer."""
    filled = np.argwhere(g > 0)
    lo, hi = filled.min(0), filled.max(0) + 1
    return g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()


def sample(seed: int, family: str | None = None) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    name = family or str(rng.choice(list(FAMILIES)))
    return name, _trim(FAMILIES[name](rng))
