"""Hand-finished 747 for the showcase (not pipeline output). The run's model is turned back into cells (its paint and
glass kept), each engine nacelle becomes a pod 8 studs wide and 11 plates tall, open at the front with a hollow
4 rows deep, the two fans the run put on the belly are taken out, the body is rebuilt by the project's builder
(mirrored brickwork), and a fan is set into the back wall of each hollow. Checked by the pipeline's checker."""
import json
import sys
from collections import Counter

import numpy as np

from samgo.builder import build
from samgo.edits import apply_edits
from samgo.free_edit_test import check
from samgo.parts import COLOR_INDEX
from samgo.primitives import cells
from samgo.rotor_data import answer

SRC = "runs/demo_samglass/747/artifacts/20261103-030812-e7b396eb/build.json"
TECH = {"technic_brick", "axle", "bush", "liftarm"}
SPANS = [(30, 36), (25, 28)]  # left wing, along y: the inboard flap, the aileron between the engines
GLASS = "tan"  # stands in for trans_clear inside the builder's color grid
NAME = {v: k for k, v in COLOR_INDEX.items()}


def grid_of(parts, shape):
    g = np.zeros(shape, int)
    for p in parts:
        if p["type"] in TECH or "w" not in p:
            continue
        c = GLASS if p.get("color") == "trans_clear" else p.get("color", "white")
        for xyz in cells(p):
            g[xyz] = COLOR_INDEX[c]
    return g


def surfaces(body, Y, fans=0):
    """An aileron and an inboard flap on the left wing's trailing edge (the flap skill's reference build at a verified
    site), each mirrored to the right wing; kept only if the whole model passes with its surfaces swinging."""
    from samgo.flap_sites import site_answer
    from samgo.mirror import mirror_edits
    from samgo.part_layout import handoff
    from samgo.primitives import frame, unframe, unframe_box

    full, body = body, [q for q in body if "w" in q and "l" in q]  # sites are planned on the stud-grid parts
    ids = [i for i, q in enumerate(body) if 0 <= q["y"] <= 40 and q["x"] >= 15 and q["z"] >= 16
           and q["type"] in ("plate", "brick")]
    n = 0
    for lo, hi in SPANS:
        h = handoff(body, ids, {"edge": "-x", "from": lo, "to": hi}, lengths=(3, 4, 5, 6), chords=(2, 3, 4))
        if h is None:
            print("surface", (lo, hi), "no site", h and h["rows"], flush=True)
            continue
        swap, mirror, X = h["frame"]
        turned = frame(body, swap, mirror, X)
        local = site_answer(turned, h["site"], color="light_gray")
        edits = [{"op": "clear", "box": unframe_box(e["box"], swap, mirror, X)} if e["op"] == "clear"
                 else {"op": "add", "part": unframe([e["part"]], swap, mirror, X)[0]} for e in local]
        twin = mirror_edits(edits, body, Y)
        if twin is None:
            print("surface", (lo, hi), "no mirror", flush=True)
            continue
        try:
            a = apply_edits(full, edits + twin)
        except Exception as e:  # noqa: BLE001
            print("surface", (lo, hi), "edit error", e, flush=True)
            continue
        v = check(full, a.parts, a.sources, [{"name": "s", "kind": "swing", "axis": "horizontal", "degrees": 30,
                                               "count": n + 2},
                                              {"name": "fans", "kind": "rotor", "axis": "horizontal", "count": fans}],
                  stand=False)
        ok = all(x[0] for k, x in v.items() if k != "stands")
        print("surface", (lo, hi), h["rows"], ok, {k: x[1] for k, x in v.items() if not x[0]}, flush=True)
        if ok:
            full, n = a.parts, n + 2
            body = [q for q in full if "w" in q and "l" in q]
            ids = [i for i, q in enumerate(body) if 0 <= q["y"] <= 40 and q["x"] >= 15 and q["z"] >= 16
                   and q["type"] in ("plate", "brick")]
    return full


def main():
    parts = json.load(open(SRC))
    parts = parts["parts"] if isinstance(parts, dict) else parts
    X = max(p["x"] + p.get("w", 1) for p in parts if "w" in p) + 2
    Y = 92
    Z = max(p["z"] + 3 for p in parts if "w" in p) + 2
    g = grid_of(parts, (X, Y, Z))
    # the belly fans' bearing slots: filled with the color around them
    for p in parts:
        if p["type"] == "technic_brick":
            for (x, y, z) in cells(p):
                around = [g[x + dx, y + dy, z] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if g[x + dx, y + dy, z]]
                g[x, y, z] = Counter(around).most_common(1)[0][0] if around else COLOR_INDEX["white"]
    occ = g > 0
    pods = []
    for (y0, y1) in [(17, 23), (29, 38)]:  # the two engines on one wing, from the voxel profile below it
        band = np.argwhere(occ[:, y0:y1 + 1, :27])
        band = band[band[:, 0] >= 30]
        xf = int(band[:, 0].max())
        front = band[band[:, 0] >= xf - 2]
        cy, cz = front[:, 1].mean() + y0, front[:, 2].mean()
        yc, z0 = int(round(cy + 0.5)), int(round(cz - 1.75))
        color = Counter(g[tuple(c)] for c in np.argwhere(occ[xf - 4:xf + 1, y0:y1 + 1, :27])
                        if g[xf - 4 + c[0], y0 + c[1], c[2]] != COLOR_INDEX[GLASS]).most_common(1)[0][0]
        color = Counter(int(g[xf - 4 + c[0], y0 + c[1], c[2]]) for c in np.argwhere(occ[xf - 4:xf + 1, y0:y1 + 1, :27])
                        ).most_common(1)[0][0]
        color = COLOR_INDEX["dark_blue"]  # the cowlings: the livery's blue
        pods.append((xf, yc, z0, color))
        pods.append((xf, Y - yc, z0, color))  # its mirror twin (y -> 91 - y; the hollow y0..y0+3 mirrors exactly)
    for (xf, yc, z0, color) in pods:
        F = xf - 3
        g[F - 2:xf + 1, yc - 4:yc + 4, z0 - 4:z0 + 7] = color     # the pod: 8 wide, 11 plates tall, back wall 2 deep
        g[F:xf + 1, yc - 3:yc + 3, z0 - 3:z0 + 6] = 0             # the hollow (the arms sweep 1.57 studs), open in front
    for (lo, hi) in SPANS:  # level the top of each control-surface strip so both hinge halves sit at one height
        strip = [(x, y) for y in range(lo, hi + 1) for x in range(X) if occ[x, y, 16:].any()]
        xe = {y: min(x for x, yy in strip if yy == y) for y in range(lo, hi + 1) if any(yy == y for _, yy in strip)}
        cols = [(x, y) for y, x0 in xe.items() for x in range(x0, x0 + 7) if occ[x, y, 16:].any()]
        T = max(16 + int(np.nonzero(occ[x, y, 16:])[0].max()) for x, y in cols)
        for x, y in cols:
            top = 16 + int(np.nonzero(occ[x, y, 16:])[0].max())
            g[x, y, top + 1:T + 1] = g[x, y, top]
        occ = g > 0
    g[:, Y // 2:, :] = g[:, Y // 2 - 1::-1, :]  # exactly mirror-symmetric (the right half is the mirror of the left)
    print("pods", pods, "symmetric", bool((g == g[:, ::-1, :]).all()), flush=True)
    body = build(g, attempts=4).pieces  # layouts ranked by stud pull standing on the table (S3b)
    body = [dict(p, color="trans_clear") if p.get("color") == GLASS else p for p in body]
    print("body", len(body), flush=True)
    json.dump(body, open(sys.argv[1].replace(".json", "_body.json"), "w"))
    done = 0
    for (xf, yc, z0, _) in pods:
        for dz in (0, 1, -1):  # the bearing at the hollow's centre, else a plate up or down (the brickwork differs)
            edits = answer("+x", xf - 3, yc - 1, z0 + dz, color="dark_gray", arm_color="light_gray")
            # the fan: a spoked rim (24 mm) locked on the axle filling the intake, in place of the two arms
            edits = [e for e in edits if e.get("part", {}).get("type") != "liftarm"] + [
                {"op": "add", "part": {"type": "wheel", "axis": "x", "x": xf - 2, "y": yc, "z": z0 + dz,
                                       "size": "24x14", "color": "light_gray"}}]
            a = apply_edits(body, edits)
            v = check(body, a.parts, a.sources, [{"name": "fan", "kind": "rotor", "axis": "horizontal", "count": 1}],
                      stand=False)
            ok = all(v[k][0] for k in ("structure", "connected", "fan"))
            print("fan", (xf, yc, z0 + dz), ok, {k: x[1] for k, x in v.items() if not x[0]}, flush=True)
            if ok:
                body, done = a.parts, done + 1
                break
    body = surfaces(body, Y, done)
    v = check(body, body, list(range(len(body))),
              [{"name": "fans", "kind": "rotor", "axis": "horizontal", "count": done}], stand=False)
    print("final", len(body), "fans", done, {k: x for k, x in v.items()}, flush=True)
    json.dump(body, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
