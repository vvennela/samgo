"""Hand-finished 747 gear, short: the old legs off, the model lowered for 24 mm wheels, two mirrored pairs of slim legs
under the lowest part of the belly (the wing-root fairing) and a nose leg. python gear_747_short.py in.json out.json"""
import json
import sys

sys.path.insert(0, "runs/hand_finished")
import gear_747_thin  # noqa: E402
from gear_747_thin import ok  # noqa: E402
from gear_axle import strip_legs  # noqa: E402

import samgo.spark_gear as sg  # noqa: E402
from samgo.free_edit_test import check  # noqa: E402
from samgo.spark_gear import raised, spots, stands  # noqa: E402

gear_747_thin.WHEEL, sg.CLEAR = "24x14", 9
TECH = ("technic_brick", "axle", "bush", "wheel", "liftarm")  # hinges stay: the surfaces hang on them


def main():
    p = strip_legs(json.load(open(sys.argv[1])))
    keep = [q for q in p if q["type"] in TECH]
    body, lift = raised([q for q in p if q not in keep])
    f = spots(body)
    low = min(f.values())
    legs = []
    for x0 in range(78, 68, -1):                       # nose leg on the centre line
        if (x0, 46) in f:
            good, q = ok(body, x0, 46, f[(x0, 46)])
            if good:
                body = q
                legs.append(("nose", x0, f[(x0, 46)] - low))
                break
    used = []
    for x0, yc in sorted([s for s in f if f[s] == low and s[1] < 46 and (s[0], 92 - s[1]) in f
                          and f[(s[0], 92 - s[1])] == low], key=lambda s: (-s[0], s[1])):
        if any(abs(x0 - u) < 4 for u in used):
            continue
        good, q = ok(body, x0, yc, low)
        if not good:
            continue
        good, q2 = ok(q, x0, 92 - yc, low)
        if good:
            body = q2
            used.append(x0)
            legs.append(("main", x0, yc))
        if len(used) == 2:
            break
    body = body + [q | {"z": q["z"] + lift} for q in keep]  # fans and surfaces back on (they confuse the leg check)
    v = check(body, body, list(range(len(body))), [{"name": "fans", "kind": "rotor", "axis": "horizontal",
                                                     "count": 4}], stand=False)
    print("legs", legs, "stands(geom)", stands(body), {k: x for k, x in v.items() if k != "stands"}, flush=True)
    json.dump(body, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
