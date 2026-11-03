"""Hand-finished 747 gear: the model raised, the leg skill's reference legs (a column, Technic bricks, an axle, two
wheels) at the nose and under the body and wings, mirrored pairs; kept only if the whole model stands on them."""
import json
import sys

from samgo.edits import apply_edits
from samgo.free_edit_test import check
from samgo.spark_gear import raised, reference, site, stands

SPOTS = [("nose", 77, 46), ("body L", 37, 48), ("body R", 37, 43), ("wing L", 42, 52), ("wing R", 42, 39),
         ("pod L", 50, 35), ("pod R", 50, 57)]


def main():
    parts = json.load(open(sys.argv[1]))
    n_fans = sum(p["type"] == "wheel" for p in parts)
    # the fans (all the Technic parts so far) are set aside while legs are placed (the leg check counts every
    # wheel, and a fan's rim is one), then put back, raised with the model
    fans = [p for p in parts if p["type"] in ("technic_brick", "axle", "bush", "wheel")]
    parts = [p for p in parts if p not in fans]
    parts, lift = raised(parts)
    taken, legs = [], []
    for name, x, y in SPOTS:
        s = site(parts, x, y, taken)
        print(name, s, flush=True)
        if s is None:
            continue
        taken.append((s["x0"], s["yc"]))
        parts = apply_edits(parts, reference(parts, s["x0"], s["yc"], s["bottom"], color="light_gray")).parts
        legs.append(name)
    parts = parts + [p | {"z": p["z"] + lift} for p in fans]
    st = stands(parts)
    v = check(parts, parts, list(range(len(parts))), [
        {"name": "fans", "kind": "rotor", "axis": "horizontal", "count": n_fans}], stand=False)
    print("legs", legs, "lift", lift, "stands", st, {k: x for k, x in v.items()}, flush=True)
    json.dump(parts, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
