"""Hand-finished car glass (large Camry): SAM's glass came out patchy at 32 studs, so the whole window band is made
glass the way the medium Camry has it. In the band (plates ZG..ZT, x X0..X1: from the model's own SAM glass), every
body cell with air beside or above it becomes trans-clear, except the corner pillars and a middle pillar; the roof
stays paint. Cells of the model's own target that the build left empty above the window line are filled (roof: paint,
band: glass). Plates are split per cell, recoloured, and merged back into real parts. Kept only if structure,
connection and standing still pass.

    python runs/hand_finished/glass_band.py in_build.json target.npz out.json ZG ZT X0 X1 [paint]
"""
import json
import sys

import numpy as np

from samgo.consolidate import consolidate
from samgo.free_edit_test import check
from samgo.kinematics import analyse
from samgo.primitives import cells, occupancy
from samgo.spark_gear import stands

STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))


def main():
    src, tgt, dst = sys.argv[1:4]
    ZG, ZT, X0, X1 = (int(v) for v in sys.argv[4:8])
    paint = sys.argv[8] if len(sys.argv) > 8 else "red"
    p = json.load(open(src))
    p = p["parts"] if isinstance(p, dict) else p
    target = np.load(tgt)["target"]
    occ = occupancy(p)
    bodies, _ = analyse(p)
    groups = list(bodies.values()) if isinstance(bodies, dict) else list(bodies)
    frame = set(max(groups, key=len))
    want = {tuple(map(int, c)) for c in np.argwhere(target > 0)}
    filled = {c for c in want - set(occ) if c[2] >= ZG and X0 <= c[0] <= X1}  # holes above the window line
    allc = set(occ) | filled

    def pillar(x, y, z):
        lay = [c for c in allc if c[2] == z and X0 <= c[0] <= X1]
        if not lay:
            return False
        xs, ys = [c[0] for c in lay], [c[1] for c in lay]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        mid = (xmin + xmax) // 2
        side = y <= ymin + 1 or y >= ymax - 1
        return side and (x <= xmin + 1 or x >= xmax - 1 or x in (mid, mid + 1))

    def band(c):
        return ZG <= c[2] <= ZT and X0 <= c[0] <= X1

    colour = {}
    for c in filled:
        colour[c] = "trans_clear" if band(c) and not pillar(*c) else paint
    for c, i in occ.items():
        q = p[i]
        if i not in frame or q["type"] != "plate" or str(q.get("color", "")).startswith("trans"):
            continue
        if band(c) and not pillar(*c) and any((c[0] + a, c[1] + b, c[2] + d) not in allc for a, b, d in STEPS):
            colour[c] = "trans_clear"
    # re-lay the whole region (every frame plate above the window line in the band's x range, and the holes) with
    # the builder's layout, so seams are staggered and every piece is gripped from above or below
    region = {c for c, i in occ.items() if i in frame and p[i]["type"] == "plate" and c[2] >= ZG and X0 <= c[0] <= X1}
    touched = {occ[c] for c in region}
    region = {c for i in touched for c in cells(p[i])} | filled
    for c in region:
        if c not in colour:
            colour[c] = p[occ[c]].get("color", paint) if c in occ else paint
    from samgo.builder import _layout
    from samgo.parts import COLOR_INDEX

    PLACE = "tan"  # placeholder the builder knows, for glass cells (the car has no tan)
    assert not any(q.get("color") == PLACE for q in p)
    names = {v: k for k, v in COLOR_INDEX.items()}
    lo = np.min(list(region), axis=0)
    hi = np.max(list(region), axis=0) + 1
    T = np.zeros(tuple(hi - lo), dtype=np.int32)
    for c in region:
        col = colour[c]
        T[c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]] = COLOR_INDEX[PLACE if col == "trans_clear" else col]
    from samgo.primitives import loose_parts

    keep = [q for i, q in enumerate(p) if i not in touched]
    best = None
    for seed in range(int(__import__("os").environ.get("TRIES", "24"))):  # layouts differ by scan order and ties
        new = _layout(T, None if seed == 0 else np.random.default_rng(seed), bricks=False)
        for q in new:
            q.update(x=q["x"] + int(lo[0]), y=q["y"] + int(lo[1]), z=q["z"] + int(lo[2]),
                     color="trans_clear" if q["color"] == PLACE else q["color"])
        n_loose = len(loose_parts(keep + new))
        if best is None or n_loose < best[0]:
            best = (n_loose, new)
        if n_loose == 0:
            break
    print("layout tries:", seed + 1, "loose parts in the best:", best[0], flush=True)
    out = keep + best[1]
    out, saved = consolidate(out)
    v = check(out, out, list(range(len(out))), [], stand=False)
    res = {k: v[k][0] for k in ("structure", "connected") if k in v}
    st = stands(out)
    glass = sum(1 for c, i in occupancy(out).items() if out[i].get("color") == "trans_clear")
    print(f"glass cells {glass}, holes filled {len(filled)}, parts {len(p)} -> {len(out)}", res, "stands", st["passed"],
          st.get("issues", [])[:1], flush=True)
    json.dump(out, open(dst + ".try", "w"))
    if all(res.values()) and st["passed"]:
        json.dump(out, open(dst, "w"))
        print("saved", dst)


if __name__ == "__main__":
    main()
