"""Hand-finished 747 gear: the model raised, the leg skill's reference legs (a column, Technic bricks, an axle, two
wheels) at the nose and under the body and wings, mirrored pairs; kept only if the whole model stands on them."""
import json
import sys

from samgo.edits import apply_edits
from samgo.free_edit_test import check
from samgo.spark_gear import raised, reference, site, stands

SPOTS = [("nose", 77, 46), ("body", 37, 50), ("wing", 44, 55)]  # nose on the centre line; each main leg mirrored


def main():
    parts = json.load(open(sys.argv[1]))
    n_fans = sum(p["type"] == "wheel" for p in parts)
    # the fans (all the Technic parts so far) are set aside while legs are placed (the leg check counts every
    # wheel, and a fan's rim is one), then put back, raised with the model
    fans = [p for p in parts if p["type"] in ("technic_brick", "axle", "bush", "wheel")]
    parts = [p for p in parts if p not in fans]
    parts, lift = raised(parts)
    taken, legs = [], []
    from samgo.spark_gear import spots
    for name, x, y in SPOTS:
        s = site(parts, x, y, taken)
        print(name, s, flush=True)
        if s is None:
            continue
        pair = [(s["x0"], s["yc"], s["bottom"])]
        if name != "nose":  # its mirror twin across the centre line (y -> 91 - y: the 2x2 spot yc-1..yc -> 92-yc)
            twin = (s["x0"], 92 - s["yc"])
            if spots(parts).get(twin) != s["bottom"]:
                print(name, "no mirror spot", flush=True)
                continue
            pair.append((*twin, s["bottom"]))
        trial = parts
        for x0, yc, bottom in pair:
            trial = apply_edits(trial, reference(trial, x0, yc, bottom, color="light_gray")).parts
        taken += [(x0, yc) for x0, yc, _ in pair]
        parts = trial
        legs.append(name)
    parts = parts + [p | {"z": p["z"] + lift} for p in fans]
    st = stands(parts)
    v = check(parts, parts, list(range(len(parts))), [
        {"name": "fans", "kind": "rotor", "axis": "horizontal", "count": n_fans}], stand=False)
    print("legs", legs, "lift", lift, "stands", st, {k: x for k, x in v.items()}, flush=True)
    json.dump(parts, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
