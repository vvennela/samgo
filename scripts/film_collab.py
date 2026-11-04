"""The Spark + Glimmer collaboration shot, from a pipeline run's real record: the greedy body, Spark's wheel
positions, then Glimmer's own added parts (assembly/trace.json) dropping in group by group.

    PYTHONPATH=. python scripts/film_collab.py <run dir> plan.json model.json

model.json: the accepted parts (body + Glimmer's running gear). plan.json: per-part steps and arrival directions for
scripts/film_tiles.py (plan=...), the timeline of drops, and Spark's markers (rings at the wheels it located).
"""
import json
import sys
from pathlib import Path

from samgo.animation import paths
from samgo.technic import HOLE_HEIGHT, PLATE, STUD, WHEELS

# when each group lands (seconds into the shot), and how long its drop takes
TIMELINE = [("rear_bricks", 4.2, 0.8), ("rear_axle", 5.4, 0.6), ("rear_wheels", 6.1, 0.6),
            ("front_bricks", 8.4, 0.8), ("front_axle", 9.6, 0.6), ("front_wheels", 10.3, 0.6)]
RINGS_AT = (1.6, 2.4)        # Spark's markers appear (rear, front)
RINGS_OFF = 11.2


def main():
    run, plan_out, model_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    parts = json.loads((run / "assembly" / "accepted.json").read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    trace = json.loads((run / "assembly" / "trace.json").read_text())
    edges = [s["edge"] for s in trace["steps"]]
    rear, front = min(edges), max(edges)
    group = {}
    for i, p in enumerate(parts):
        if p["type"] not in ("technic_brick", "axle", "wheel"):
            continue
        side = "rear" if p["x"] < (rear + front) / 2 else "front"
        kind = {"technic_brick": "bricks", "axle": "axle", "wheel": "wheels"}[p["type"]]
        group[i] = f"{side}_{kind}"
    names = [g for g, _, _ in TIMELINE]
    steps = [names.index(group[i]) + 1 if i in group else 0 for i in range(len(parts))]
    dirs, blocked = paths(parts, steps)
    wheel = next(p for p in parts if p["type"] == "wheel")
    r = WHEELS[wheel.get("size", "30.4x14")][0] / 2
    z = wheel["z"] * PLATE + HOLE_HEIGHT
    width = max(p["y"] + p.get("l", 1) for p in parts if "l" in p) * STUD
    rings = []
    for k, edge in enumerate((rear, front)):
        for side in (-6.0, width + 6.0):   # both sides of the car, just outside the body
            rings.append({"x": edge * STUD, "up": z, "z": side, "r": r + 3, "t0": RINGS_AT[k], "t1": RINGS_OFF})
    plan = {"steps": steps, "dirs": [[d[0], d[2], d[1]] for d in dirs],
            "events": [{"step": names.index(g) + 1, "t": t, "dur": d} for g, t, d in TIMELINE],
            "rings": rings, "blocked": blocked}
    plan_out.write_text(json.dumps(plan))
    model_out.write_text(json.dumps(parts))
    print(f"{len(parts)} parts, groups {sorted(set(group.values()))}, blocked {blocked}")


if __name__ == "__main__":
    main()
