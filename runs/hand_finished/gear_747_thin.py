"""Hand-finished 747 gear with slim legs: one Technic brick for the axle and a 1-stud-thick column (2x1 bricks)
up to the underside, in place of the leg skill's 2x2 column; nose leg plus two mirrored main pairs."""
import json
import sys

from samgo.edits import apply_edits
from samgo.free_edit_test import check
from samgo.spark_gear import LEG, raised, spots


WHEEL = "30.4x14"


def leg(x0, yc, bottom, color="light_gray"):
    from samgo.technic import WHEELS, HOLE_HEIGHT, PLATE
    za = round((WHEELS[WHEEL][0] / 2 - HOLE_HEIGHT) / PLATE)   # the axle's brick, so the tyre touches the table
    adds, z = [], za + 3
    while z + 3 <= bottom:
        adds.append(dict(type="brick", w=2, l=1, x=x0, y=yc - 1, z=z, color=color))
        z += 3
    while z < bottom:
        adds.append(dict(type="plate", w=2, l=1, x=x0, y=yc - 1, z=z, color=color))
        z += 1
    xg = x0 + 1
    adds += [dict(type="technic_brick", w=2, l=1, x=x0, y=yc - 1, z=za, color=color, hole="round"),
             dict(type="axle", axis="y", x=xg, z=za, y=yc - 4, length=8),
             dict(type="wheel", axis="y", x=xg, z=za, y=yc - 3, size=WHEEL, color="black"),
             dict(type="wheel", axis="y", x=xg, z=za, y=yc + 1, size=WHEEL, color="black"),
             dict(type="bush", axis="y", x=xg, z=za, y=yc - 4), dict(type="bush", axis="y", x=xg, z=za, y=yc + 3)]
    return [{"op": "add", "part": a} for a in adds]


def ok(parts, x0, yc, bottom):
    a = apply_edits(parts, leg(x0, yc, bottom))
    v = check(parts, a.parts, a.sources, LEG, stand=False)
    return all(v[k][0] for k in ("structure", "connected", "leg")), a.parts


def main():
    p = json.load(open(sys.argv[1]))
    fans = [q for q in p if q["type"] in ("technic_brick", "axle", "bush", "wheel")]
    body, lift = raised([q for q in p if q not in fans])
    f = spots(body)
    legs = []
    for x0 in range(77, 70, -1):
        if (x0, 46) in f:
            good, q = ok(body, x0, 46, f[(x0, 46)])
            if good:
                body = q
                legs.append(("nose", x0, 46))
                break
    for name, band in (("body", range(51, 55)), ("wing", range(57, 64))):
        cand = sorted([s for s in f if s[1] in band and (s[0], 92 - s[1]) in f and f[s] == f[(s[0], 92 - s[1])]
                       and 30 <= s[0] <= 50], key=lambda s: abs(s[0] - 40))
        for x0, yc in cand[:20]:
            good, q = ok(body, x0, yc, f[(x0, yc)])
            if not good:
                continue
            good, q2 = ok(q, x0, 92 - yc, f[(x0, yc)])
            if good:
                body = q2
                legs.append((name, x0, yc))
                break
    body = body + [q | {"z": q["z"] + lift} for q in fans]
    v = check(body, body, list(range(len(body))), [{"name": "fans", "kind": "rotor", "axis": "horizontal",
                                                     "count": 4}], stand=False)
    print("legs", legs, {k: x for k, x in v.items() if k != "stands"}, flush=True)
    json.dump(body, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
