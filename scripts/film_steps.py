"""The certified build as seamless stop motion: a plan for scripts/film_tiles.py (plan=...) that drops each step of
samgo.manual's certified order onto the model, accelerating, with no pages or callouts.

    PYTHONPATH=. python scripts/film_steps.py build.json manual.json plan.json [seconds]
"""
import json
import sys
from pathlib import Path

from samgo.animation import paths


def main():
    parts = json.loads(Path(sys.argv[1]).read_text())
    parts = parts["parts"] if isinstance(parts, dict) else parts
    manual = json.loads(Path(sys.argv[2]).read_text())
    total = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    steps = [0] * len(parts)
    for k, s in enumerate(manual["steps"]):
        for i in s["parts"]:
            steps[i] = k + 1
    dirs, blocked = paths(parts, steps)
    n = len(manual["steps"])
    # accelerating: step k lasts d_k, shrinking geometrically; scaled to the total
    d, ds = 1.0, []
    for _ in range(n):
        ds.append(d); d *= 0.95
    scale = (total - 0.6) / sum(ds)
    t, events = 0.6, []
    for k, dk in enumerate(ds):
        dur = min(0.35, dk * scale * 0.9)
        t += dk * scale
        events.append({"step": k + 1, "t": round(t, 3), "dur": round(dur, 3)})
    Path(sys.argv[3]).write_text(json.dumps({"steps": steps, "dirs": [[x[0], x[2], x[1]] for x in dirs],
                                             "events": events, "rings": [], "blocked": blocked}))
    print(f"{n} steps over {total}s, blocked {blocked}")


if __name__ == "__main__":
    main()
