"""Granularity sweep: how much of the model Glimmer sees, and how exactly it is told where, against success,
answer size and time - for one mechanism at a time, with the same generic checks at every level.

Levels (prompts are hand-written per level on purpose: this measures where the knowing/doing handoff should sit):
  L0 the whole object; L1 one large structure (a region from samgo.regions); L2 a small area of it; L3 the exact
  line or position. Runs are independent and go in parallel.

    python -m samgo.granularity runs/s6/boeing747 --model <sampler path> --out runs/granularity/r5.json
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .edits import apply_edits
from .free_edit_test import RULES, check, explain
from .running_gear_task import FEEDBACK, edits_from

H = {"plate": 1, "brick": 3}


def setup(folder: Path):
    body = json.loads((folder / "body40.json").read_text())
    labels = np.load(folder / "regions40.npy")

    def region_of(p):
        cells = labels[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["l"], p["z"]:p["z"] + H[p["type"]]].ravel()
        cells = cells[cells > 0]
        return int(np.bincount(cells).argmax()) if cells.size else 0

    return body, [region_of(p) for p in body]


def tasks(body, regions):
    """(mechanism, level, local part indices, instruction, requirements) for the sweep."""
    allp = list(range(len(body)))
    wing = [i for i in allp if regions[i] == 2]
    # the wing's trailing edge: its rearmost cells per row; the strip is the last 4 studs of chord
    te = {}
    for i in wing:
        p = body[i]
        for y in range(p["y"], p["y"] + p["l"]):
            te[y] = max(te.get(y, -1), p["x"] + p["w"])
    rows = sorted(te)
    span = rows[len(rows) // 4: len(rows) // 2] or rows          # an inboard stretch of the span
    xh = min(te[y] for y in span) - 3
    strip = [i for i in wing if body[i]["x"] + body[i]["w"] > xh - 2 and
             any(y in span for y in range(body[i]["y"], body[i]["y"] + body[i]["l"]))]
    zs = max(body[i]["z"] for i in strip) if strip else 0
    fus = [i for i in allp if regions[i] == 1]
    zmid = float(np.median([body[i]["z"] for i in fus]))
    lower = [i for i in fus if body[i]["z"] <= zmid]
    xs = [body[i]["x"] for i in lower]
    xg = int(np.median(xs))
    yc = int(np.median([body[i]["y"] for i in fus]))
    patch = [i for i in lower if abs(body[i]["x"] - xg) <= 4 and abs(body[i]["y"] - yc) <= 3]
    zb = min(body[i]["z"] for i in patch) if patch else 0
    flap = [{"name": "flap", "kind": "swing", "axis": "horizontal", "degrees": 20, "count": 1}]
    gear = [{"name": "gear", "kind": "spin", "count": 2}]
    return [
        ("flap", "L0", allp, "This is a Boeing 747. Add one flap on a wing's trailing edge: free a strip of the "
                             "trailing edge and hinge it so it swings about a horizontal axis.", flap),
        ("flap", "L1", wing, "These are the parts of one wing of a Boeing 747. Add one flap on its trailing edge: "
                             "free a strip of the trailing edge and hinge it about a horizontal axis.", flap),
        ("flap", "L2", strip, "These are the parts along the trailing edge of a wing. Turn the rear of them into a "
                              "flap: free it and hinge it about a horizontal axis along the edge.", flap),
        ("flap", "L3", strip, f"These are the parts along the trailing edge of a wing. Make a flap behind the line "
                              f"x = {xh} (rows y = {span[0]}..{span[-1]}): remove what is at x = {xh} to leave a "
                              f"one-stud gap, keep the parts behind it as the flap, and hinge it with bar plates "
                              f"(facing +x, ending at x = {xh}) and clip plates (facing -x, starting at x = {xh + 1}) "
                              f"at z = {zs}.", flap),
        ("gear", "L0", allp, "This is a Boeing 747. Add one landing-gear leg with an axle and two wheels that spin; "
                             "the wheels must be the lowest parts.", gear),
        ("gear", "L1", lower, "These are the lower parts of the fuselage of a Boeing 747. Add one landing-gear leg "
                              "under it with an axle and two wheels that spin; the wheels must be the lowest parts.",
         gear),
        ("gear", "L2", patch, "These are the belly parts of an airplane around one spot. Add a landing-gear leg "
                              "here: a column down from the belly, an axle and two wheels that spin, lowest of all.",
         gear),
        ("gear", "L3", patch, f"These are the belly parts of an airplane around one spot. Add a landing-gear leg: a "
                              f"2x2 column of bricks from the belly down to two 1x2 Technic bricks (w=2, l=1) at "
                              f"x = {xg - 1}, y = {yc - 1} and y = {yc}, z = 0; an axle along y on stud edge "
                              f"x = {xg}, z = 0, from y = {yc - 4}, length 10; wheels at y = {yc - 3} and "
                              f"y = {yc + 1}; bushes outboard. The column must reach the belly bottom at "
                              f"z = {zb}.", gear),
    ]


def run_one(body, task, model, rounds=3, max_tokens=12000, precise=True):
    from .assembly import FireworksSampler

    name, level, local, instruction, reqs = task
    sampler = FireworksSampler(model)
    chat = [{"role": "system", "content": RULES},
            {"role": "user", "content": instruction + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(body[i])}" for k, i in enumerate(local))}]
    record = {"mechanism": name, "level": level, "parts_shown": len(local), "rounds": []}
    last = None
    for rnd in range(1, rounds + 1):
        t = time.time()
        entry = {"round": rnd}
        try:
            s = sampler(chat, max_tokens=max_tokens, temperature=0.0 if rnd == 1 else 0.7, timeout=600)
            entry.update(tokens=s.tokens, seconds=round(time.time() - t, 1))
            edits = edits_from(s.text)
            entry["repeat"] = s.text == last
            last = s.text
            a = apply_edits(body, edits, local=local)
            v = check(body, a.parts, a.sources, reqs)
            keep = {k: v[k] for k in v if k in ("structure", "connected", "gear") or k.endswith("surfaces")}
            entry.update(edits=len(edits), valid=True, passed=all(x[0] for x in keep.values()),
                         ops={op: sum(e["op"] == op for e in edits) for op in ("remove", "set", "add")},
                         checks={k: [bool(x[0]), str(x[1])[:100]] for k, x in keep.items()})
            issues = (explain(body, a.parts, a.sources, a.changed, keep, local) if precise
                      else [f"{k}: {x[1]}" for k, x in keep.items() if not x[0]])
            entry["feedback"] = issues
        except Exception as e:  # noqa: BLE001
            keep = {"answer": (False, f"{type(e).__name__}: {e}"[:150])}
            issues = [f"answer: {keep['answer'][1]}"]
            entry.update(valid=False, passed=False, checks={"answer": [False, keep["answer"][1]]})
            s = locals().get("s")
        record["rounds"].append(entry)
        if entry["passed"]:
            break
        if s is not None:
            chat = chat + [{"role": "assistant", "content": s.text},
                           {"role": "user", "content": FEEDBACK.format(
                               issues="\n".join(f"- {i}" for i in issues))}]
    record["passed"] = any(r["passed"] for r in record["rounds"])
    return record


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coarse-feedback", action="store_true", help="the old check-level feedback, for comparison")
    args = ap.parse_args()
    body, regions = setup(args.folder)
    work = tasks(body, regions)
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        results = list(pool.map(lambda t: run_one(body, t, args.model, precise=not args.coarse_feedback), work))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    for r in results:
        first = r["rounds"][0]
        print(f"{r['mechanism']:5s} {r['level']}  shown {r['parts_shown']:3d}  passed {r['passed']!s:5s}  "
              f"first: {first.get('tokens', '-')} tokens, {first.get('seconds', '-')} s, valid {first['valid']}  "
              f"| {[c for c, x in first['checks'].items() if not x[0]]}")


if __name__ == "__main__":
    main()
