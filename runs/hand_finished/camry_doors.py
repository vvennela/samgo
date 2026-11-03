"""Hand-finished Camry doors: the side made flat between the wheel arches (outer row y=0 / y=10 filled from the sill to
the roof line, the skin row behind it opened), then the project's door builder cuts a door there; the door's top rows
are glass (the window moves with the door). Kept only if every car check passes."""
import json
import sys

from samgo.consolidate import consolidate
from samgo.doors import add_car_doors
from samgo.pipeline import _door_reports
from samgo.primitives import occupancy

sys.path.insert(0, "runs/hand_finished")
from hollow_camry import hollow  # noqa: E402

SRC = "runs/demo_r10/camry_samglass/artifacts/20261103-025644-d86cb91e/build.json"


def main():
    X0, X1, Z0, Z1, ZG = [int(v) for v in sys.argv[2:7]]  # door span, bottom/top plate, first glass plate
    src = sys.argv[7] if len(sys.argv) > 7 else SRC
    paint = sys.argv[8] if len(sys.argv) > 8 else "red"
    p = json.load(open(src))
    p = p["parts"] if isinstance(p, dict) else p
    Y = max(q["y"] + q["l"] for q in p if "l" in q) - 1          # the far side row
    for row, inner in ((0, 1), (Y, Y - 1)):
        p = hollow(p, (X0, X1, inner, inner, Z0 + 1, Z1))           # the skin behind the door opens
        occ = occupancy(p)
        p += [{"type": "plate", "w": 1, "l": 1, "x": x, "y": row, "z": z,
               "color": "trans_clear" if z >= ZG else paint}
              for x in range(X0, X1 + 1) for z in range(Z0, Z1 + 1) if (x, row, z) not in occ]
    p, _ = consolidate(p)
    p = [dict(q, color="tan") if q.get("color") == "trans_clear" else q for q in p]
    r = add_car_doors(p, outline={"x0": X0, "x1": X1, "z0": Z0, "z1": Z1}, front_high=True)
    q = r["parts"]
    doors = r.get("doors", [])
    # the door's own rows from ZG up are its window
    from samgo.learned_mechanisms import car_door_context
    moving = {i for g, _ in car_door_context(q)["moving"].values() for i in g}
    def colour(i, x):
        if i in moving and x["type"] in ("plate", "brick"):  # the door: its window from ZG up, paint below
            return dict(x, color="trans_clear" if x["z"] >= ZG else paint)
        if x.get("color") == "tan":
            return dict(x, color="trans_clear")
        if x.get("color") == "dark_gray" and x["type"] in ("plate", "brick"):  # hinge holders show: body paint
            return dict(x, color=paint)
        return x
    q = [colour(i, x) for i, x in enumerate(q)]
    ok, rep = _door_reports(q)
    print("doors", [(d["side"], d["x"], d["z"]) for d in doors], "passed", ok,
          {k: v.get("passed") for k, v in rep.items() if isinstance(v, dict) and "passed" in v}, flush=True)
    print({k: v for k, v in rep["checks"].items()}, rep["doors"], flush=True)
    if ok:
        json.dump(q, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
