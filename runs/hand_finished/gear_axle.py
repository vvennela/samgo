"""Hand-finishing gear the way LEGO planes are built: no columns. Main gear = two 1x2 Technic bricks under the belly,
one at each edge of it, one axle through both and a wheel (with a bush) outside each; the belly sits on the bricks.
Nose gear = the slim leg (gear_747_thin.leg): one Technic brick plus a short 2x1 column up to the nose's underside.
The model is set so the main bricks' tops meet the belly: the gear is as short as the wheels allow.
Kept only if structure, connection, the leg check and standing pass.

    python gear_axle.py in.json out.json main_x y0 y1 nose_x nose_yc [wheel]
    (main bricks at x main_x..main_x+1, at rows y0 and y1; nose leg at nose_x, rows nose_yc-1..nose_yc)
"""
import json
import sys

sys.path.insert(0, "runs/hand_finished")
import gear_747_thin  # noqa: E402
from gear_747_thin import ok  # noqa: E402

from samgo.free_edit_test import check  # noqa: E402
from samgo.primitives import occupancy  # noqa: E402
from samgo.spark_gear import stands  # noqa: E402
from samgo.technic import HOLE_HEIGHT, PLATE, WHEELS  # noqa: E402

GRID = ("plate", "brick", "slope")


def strip_legs(p):
    """Remove slim legs (light gray Technic brick at the bottom, its 2x1 column, axle, wheels, bushes)."""
    za = min((q["z"] for q in p if q["type"] == "technic_brick"), default=None)
    legs = [q for q in p if q["type"] == "technic_brick" and q["z"] == za and q.get("color") == "light_gray"]
    drop = set()
    for b in legs:
        drop.add(id(b))
        for q in p:
            if q["type"] in GRID and q.get("color") == "light_gray" and (q["x"], q["y"], q["w"], q["l"]) == (b["x"], b["y"], 2, 1):
                drop.add(id(q))
            if q["type"] in ("axle", "wheel", "bush") and q.get("axis") == "y" and q["x"] == b["x"] + 1 and q["z"] == za:
                drop.add(id(q))
    return [q for q in p if id(q) not in drop]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    mx, y0, y1, nx, nyc = (int(v) for v in sys.argv[3:8])
    if len(sys.argv) > 8:
        gear_747_thin.WHEEL = sys.argv[8]
    wheel = gear_747_thin.WHEEL
    p = strip_legs(json.load(open(src)))
    occ = occupancy([q for q in p if q["type"] in GRID])
    belly = {}
    for x, y, z in occ:
        belly[(x, y)] = min(belly.get((x, y), 99), z)
    cells = [(x, y) for x in (mx, mx + 1) for y in (y0, y1)]
    if len({belly.get(c) for c in cells}) != 1 or None in {belly.get(c) for c in cells}:
        sys.exit(f"the belly is not flat under the main bricks: {[belly.get(c) for c in cells]}")
    za = round((WHEELS[wheel][0] / 2 - HOLE_HEIGHT) / PLATE)  # the axle brick, so the tyre touches the table
    shift = za + 3 - belly[cells[0]]                            # the belly sits on the bricks
    p = [q | {"z": q["z"] + shift} for q in p]
    body = [q for q in p if q["type"] in GRID]
    if min(q["z"] for q in body) < za + 3 - 0:
        low = min(q["z"] for q in body)
        print(f"note: some of the body reaches z={low}, below the belly at {za + 3}", flush=True)
    xg = mx + 1
    n = y1 - y0 + 1
    main_gear = [dict(type="technic_brick", w=2, l=1, x=mx, y=y, z=za, color="light_gray", hole="round") for y in (y0, y1)]
    main_gear += [dict(type="axle", axis="y", x=xg, z=za, y=y0 - 3, length=n + 6, color="light_gray"),
                  dict(type="wheel", axis="y", x=xg, z=za, y=y0 - 2, size=wheel, color="black"),
                  dict(type="wheel", axis="y", x=xg, z=za, y=y1 + 1, size=wheel, color="black"),
                  dict(type="bush", axis="y", x=xg, z=za, y=y0 - 3, color="light_gray"),
                  dict(type="bush", axis="y", x=xg, z=za, y=y1 + 3, color="light_gray")]
    p = p + main_gear
    occ = occupancy([q for q in p if q["type"] in GRID])
    nose_bottom = min(z for x, y, z in occ if (x, y) in ((nx, nyc - 1), (nx + 1, nyc - 1), (nx, nyc), (nx + 1, nyc)))
    good, q = ok(p, nx, nyc, nose_bottom)
    print("nose leg ok", good, "length (plates)", nose_bottom - za - 3, flush=True)
    if good:
        p = q
    st = stands(p)
    v = check(p, p, list(range(len(p))), [], stand=False)
    print("stands", st, {k: x for k, x in v.items() if k != "stands"}, flush=True)
    json.dump(p, open(dst, "w"))


if __name__ == "__main__":
    main()
