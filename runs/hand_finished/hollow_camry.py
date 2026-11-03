"""Hand-finished Camry cabin: the interior emptied (inside the walls, above the floor, under the roof), parts that
straddle the boundary cut back to 1x1 plates outside it and merged into real pieces; kept only if the car still
passes every check (one structure, stands on its wheels, doors swing, wheels spin)."""
import json
import sys

from samgo.consolidate import consolidate
from samgo.pipeline import _door_reports
from samgo.primitives import cells


def hollow(parts, box):
    x0, x1, y0, y1, z0, z1 = box
    inside = lambda c: x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and z0 <= c[2] <= z1
    keep, refill = [], []
    for p in parts:
        if p["type"] not in ("plate", "brick") or "w" not in p:
            keep.append(p)
            continue
        cs = cells(p)
        if not any(inside(c) for c in cs):
            keep.append(p)
            continue
        refill += [{"type": "plate", "w": 1, "l": 1, "x": x, "y": y, "z": z, "color": p.get("color", "red")}
                   for (x, y, z) in cs if not inside((x, y, z))]
    out, _ = consolidate(keep + refill)
    return out


def main():
    parts = json.load(open(sys.argv[1]))
    best = None
    for box in [(4, 13, 2, 8, 8, 13), (5, 12, 2, 8, 8, 13), (5, 12, 2, 8, 9, 13), (6, 12, 3, 7, 9, 13)]:
        q = hollow(parts, box)
        ok, rep = _door_reports(q)
        print(box, len(q), ok, {k: v.get("passed") for k, v in rep.items() if isinstance(v, dict) and "passed" in v},
              flush=True)
        if ok:
            best = q
            break
    if best:
        json.dump(best, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
