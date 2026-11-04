"""Plates outside, bricks inside: the visible shell keeps its plates (the fine shape), the hidden inside is re-laid
with bricks where three layers of it are free (one brick locks three layers together, and is one piece instead of
three or more).

A cell is shell if any of its six neighbours is air (no part there, or outside the model). Only plates of the main
rigid body whose cells are all inside are re-laid; doors and other moving parts, glass, Technic, hinges and every
plate that shows anywhere stay exactly as they are. The inside is one colour (it cannot be seen): the most common
colour of the plates it replaces. Kept only if the model still passes the caller's checks.

    python -m samgo.inner_bricks in.json out.json
"""

import json
import sys
from collections import Counter

import numpy as np

from .parts import COLOR_INDEX

COLOR_NAMES = {i: n for n, i in COLOR_INDEX.items()}
STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def inner_bricks(parts: list[dict]) -> tuple[list[dict], dict]:
    from .builder import _layout
    from .kinematics import analyse
    from .primitives import cells, occupancy

    occ = occupancy(parts)
    bodies, _ = analyse(parts)
    groups = list(bodies.values()) if isinstance(bodies, dict) else list(bodies)
    frame = set(max(groups, key=len))  # the main rigid body; everything that moves is in another one
    shell = {c for c in occ if any((c[0] + a, c[1] + b, c[2] + d) not in occ for a, b, d in STEPS)}
    inner = [i for i, p in enumerate(parts) if i in frame and p["type"] == "plate"
             and not str(p.get("color", "")).startswith("trans")
             and all(c not in shell for c in cells(p))]
    if not inner:
        return parts, {"replaced": 0, "added": 0, "bricks": 0}
    freed = [c for i in inner for c in cells(parts[i])]
    color = Counter(parts[i].get("color", "light_gray") for i in inner).most_common(1)[0][0]
    lo = np.min(freed, axis=0)
    hi = np.max(freed, axis=0) + 1
    target = np.zeros(tuple(hi - lo), dtype=np.int32)
    for x, y, z in freed:
        target[x - lo[0], y - lo[1], z - lo[2]] = COLOR_INDEX[color]
    new = _layout(target, None, bricks=True)
    for p in new:
        p.update(x=p["x"] + int(lo[0]), y=p["y"] + int(lo[1]), z=p["z"] + int(lo[2]), color=COLOR_NAMES[COLOR_INDEX[color]])
    drop = set(inner)
    out = [p for i, p in enumerate(parts) if i not in drop] + new
    return out, {"replaced": len(inner), "added": len(new), "bricks": sum(p["type"] == "brick" for p in new),
                 "color": color}


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    p = json.load(open(src))
    p = p["parts"] if isinstance(p, dict) else p
    out, info = inner_bricks(p)
    print(json.dumps(info), len(p), "->", len(out))
    json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    main()
