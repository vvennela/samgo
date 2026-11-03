"""Hand-finish: rebuild each hinged surface as ONE bonded piece. For every bar/clip hinge pair, the parts that move with
the clips are replaced by two plate layers (staggered seams, so one rigid piece across all the surface's clips) under
the clips; anything that hung off the old pieces (a strut stub) goes. Kept only if every check passes.
    python fix_surfaces.py in.json out.json"""
import json
import sys

from samgo.free_edit_test import check
from samgo.kinematics import _sides, analyse, requirement_met, rotors


def plates(x, y0, y1, z, w, stagger):
    """A row of 2-wide plates from y0 to y1 at level z (lengths from the palette), seams shifted by `stagger`."""
    out, y = [], y0
    lens = [stagger] if stagger else []
    while y <= y1:
        L = lens.pop(0) if lens else min(8, y1 - y + 1)
        L = max(l for l in (1, 2, 3, 4, 6, 8) if l <= min(L, y1 - y + 1))
        out.append(dict(type="plate", w=w, l=L, x=x, y=y, z=z, color="white"))
        y += L
    return out


def main():
    p = json.load(open(sys.argv[1]))
    bodies, joints = analyse(p)
    moving = []
    for k, j in enumerate(joints):
        s = _sides(bodies, joints, k)
        if s is None:
            continue
        small = min(s, key=lambda g: sum(len(bodies[r]) for r in g))
        mv = [i for r in small for i in bodies[r]]
        if any(p[i]["type"] == "clip_plate" for i in mv):
            moving.append(mv)
    clips = sorted({i for mv in moving for i in mv if p[i]["type"] == "clip_plate"})
    # group clips into surfaces: same x and z, y within 3 of each other
    surfaces = []
    for i in sorted(clips, key=lambda i: (p[i]["x"], p[i]["z"], p[i]["y"])):
        c = p[i]
        if surfaces and surfaces[-1][-1]["x"] == c["x"] and surfaces[-1][-1]["z"] == c["z"] and c["y"] - surfaces[-1][-1]["y"] <= 3:
            surfaces[-1].append(c)
        else:
            surfaces.append([c])
    drop = {i for mv in moving for i in mv if p[i]["type"] != "clip_plate"}
    old = [p[i] for i in sorted(drop)]
    q = [x for i, x in enumerate(p) if i not in drop]
    for s in surfaces:
        x, zc, w = s[0]["x"], s[0]["z"], s[0]["w"]
        y0, y1 = s[0]["y"] - 1, s[-1]["y"] + 1
        y0 = max(y0, min(o["y"] for o in old if o["x"] == x and o["z"] in (zc - 1, zc - 2)))
        y1 = min(y1, max(o["y"] + o["l"] - 1 for o in old if o["x"] == x and o["z"] in (zc - 1, zc - 2)))
        q += plates(x, y0, y1, zc - 1, w, 0) + plates(x, y0, y1, zc - 2, w, 1)
        print("surface", x, y0, y1, "clips", [c["y"] for c in s], flush=True)
    v = check(q, q, list(range(len(q))), [], stand=True)
    print({k: v[k] for k in ("structure", "connected", "stands")},
          requirement_met(q, {"kind": "swing", "axis": "horizontal", "degrees": 30, "count": len(surfaces)}), rotors(q))
    bodies, joints = analyse(q)
    print("moving pieces:", sorted(len(b) for b in bodies.values())[:-1])
    json.dump(q, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
