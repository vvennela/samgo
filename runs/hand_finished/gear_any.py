"""Hand-finishing gear for any raised model: slim legs (gear_747_thin.leg), one on the centre line near the front and
mirrored pairs near the middle; kept as a set only if structure, connection and standing all pass.
    python gear_any.py in.json out.json front_x centre_y main_x"""
import json
import sys

sys.path.insert(0, "runs/hand_finished")
from gear_747_thin import ok  # noqa: E402

from samgo.free_edit_test import check  # noqa: E402
from samgo.spark_gear import raised, spots, stands  # noqa: E402

TECH = ("axle", "bush", "wheel", "propeller", "liftarm")  # bearings stay: they are part of the structure


def main():
    src, dst, fx, cy2, mx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    if len(sys.argv) > 7:  # smaller wheels, lower lift: "24x14 9"
        import gear_747_thin
        import samgo.spark_gear as sg
        gear_747_thin.WHEEL, sg.CLEAR = sys.argv[6], int(sys.argv[7])
    p = json.load(open(src))
    keep = [q for q in p if q["type"] in TECH]          # mechanisms set aside, put back raised
    body, lift = raised([q for q in p if q not in keep])
    f = spots(body)
    legs = []
    nose = sorted([s for s in f if 2 * s[1] - 1 == cy2], key=lambda s: abs(s[0] - fx))  # spot yc-1..yc centred
    for x0, yc in nose[:15]:
        good, q = ok(body, x0, yc, f[(x0, yc)])
        if good:
            body = q
            legs.append(("nose", x0, yc))
            break
    for dy in range(3, 12):
        cand = sorted([s for s in f if 2 * s[1] - 1 - cy2 == 2 * dy and (s[0], cy2 + 1 - s[1] + 0) in f], key=lambda s: abs(s[0] - mx))
        done = False
        for x0, yc in cand[:10]:
            twin = (x0, cy2 + 1 - yc)  # mirror of the spot yc-1..yc about the centre line cy2/2
            if f.get(twin) != f[(x0, yc)]:
                continue
            good, q = ok(body, x0, yc, f[(x0, yc)])
            if not good:
                continue
            good, q2 = ok(q, *twin, f[twin])
            if good:
                body, done = q2, True
                legs.append(("main", x0, yc))
                break
        if done:
            break
    body = body + [q | {"z": q["z"] + lift} for q in keep]
    st = stands(body)
    v = check(body, body, list(range(len(body))), [], stand=False)
    print("legs", legs, "stands", st, {k: x for k, x in v.items() if k != "stands"}, flush=True)
    json.dump(body, open(dst, "w"))


if __name__ == "__main__":
    main()
