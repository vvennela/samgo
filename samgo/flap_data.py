"""Run 8 training data: flaps on thin structures that look like real voxelized ones.

Run 7's sheets were flat slabs of even thickness lying on a block. Real models out of SAM 3D + voxelize are not:
a wing is thick at the root and thin at the trailing edge, its edges are ragged, its top steps (dihedral, uneven
thickness), it can leave the body at mid-height, and things hang under it. Every layer of a voxel model is tiled
with plates, so these bodies are too: a voxel set, each z layer tiled with real plates, alternate layers crossed.

Every task's answer is a verified site from samgo.flap_sites (a plain answer there passes the generic checks), so
the data teaches only generic moves: clear a channel on three sides, put the bar and clip plates *on top* of the
sheet at the hinge line (one plate above the local top), never inside it. Nothing here is fit to a test object.

Rows: the reference answer; or a wrong answer (the habitual mistakes - hinge plates inside the top layer, a
missing side channel, a hinge a stud off), the precise feedback for it, then the fix. Glimmer's own wrong answers
are added by samgo.flap_mistakes.

    python -m samgo.flap_data --out data/flaps_r8 --test runs/audit/r8-wings
"""

import argparse
import copy
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .edits import apply_edits
from .flap_sites import FLAP, site_answer, sites
from .free_edit_test import RULES, check, explain
from .primitives import COLORS, loose_parts, tile_cells, unframe, unframe_box
from .running_gear_task import FEEDBACK

KEEP = ("structure", "connected")
OBJECTS = ("airplane", "glider", "jet", "drone", "spaceship", "seaplane", "bird", "model aircraft")


# ---------------------------------------------------------------- bodies

def wing_body(rng: random.Random) -> tuple[list[dict], set]:
    """A fuselage-like block with a wing (span along +y, chord along x, trailing edge toward +x), sometimes its
    mirror on the -y side; plates only. Returns (parts, ids of the wing's parts outside the block)."""
    from .verifier import verify

    for _ in range(30):
        cells: set = set()
        bx, by, bz = rng.randint(10, 22), rng.randint(3, 6), rng.randint(5, 12)
        ox, oy = 2, 2 + 14
        for x in range(ox, ox + bx):
            for y in range(oy, oy + by):
                for z in range(0, bz):
                    cells.add((x, y, z))
        reach = rng.randint(6, 14)
        c_root, c_tip = rng.randint(6, min(12, bx)), rng.randint(3, 6)
        t_root, t_tip = rng.randint(3, 6), 2
        sweep = rng.choice((0, 1, 2, 3, 4))
        zmid0 = rng.choice((bz - 1, bz, rng.randint(2, max(2, bz - 2))))  # on top, or out of the side mid-height
        dihedral = rng.choice((0, 0, 4, 6, 8))  # one plate up every this many rows (0: flat)
        lead0 = ox + rng.randint(0, max(0, bx - c_root))
        pod = rng.random() < 0.4
        pod_y = rng.randint(2, max(2, reach - 3))
        jag = rng.choice((0.0, 0.25, 0.5))
        wing = set()
        for v in range(-2, reach):  # two rows over the block, the rest sticking out
            t = max(v, 0)
            f = t / max(reach - 1, 1)
            chord = round(c_root + (c_tip - c_root) * f)
            x0 = lead0 + round(sweep * f * 2) + (rng.choice((-1, 1)) if rng.random() < jag else 0)
            x1 = x0 + chord + (rng.choice((-1, 1)) if rng.random() < jag else 0)
            zmid = zmid0 + (t // dihedral if dihedral else 0)
            span_t = t_root + (t_tip - t_root) * f
            for x in range(x0, x1):
                c = (x - x0) / max(x1 - x0 - 1, 1)  # 0 leading edge .. 1 trailing edge
                th = max(2, round(span_t * (1.0 - 0.6 * max(0.0, c - 0.3) / 0.7)
                                  + (rng.choice((-1, 0, 0, 1)) if jag else 0)))
                z0 = zmid - th // 2
                for z in range(max(z0, 0), z0 + th):
                    wing.add((x, oy + by - 1 + v, z))
            if pod and v == pod_y:  # a pod hanging under the wing (engine, float, tank)
                px = x0 + rng.randint(0, max(0, chord // 3))
                for x in range(px - 2, px + rng.randint(3, 6)):
                    for y in (oy + by - 1 + v, oy + by + v):
                        for z in range(max(zmid - th // 2 - rng.randint(2, 4), 1), zmid - th // 2):
                            wing.add((x, y, z))
        both = rng.random() < 0.5
        mirrored = {(x, 2 * oy + by - 1 - y, z) for (x, y, z) in wing} if both else set()
        cells |= wing | mirrored
        if min(c[0] for c in cells) < 0:
            continue
        color = rng.choice(COLORS)
        parts = []
        for z in sorted({c[2] for c in cells}):
            layer = {(x, y) for (x, y, zz) in cells if zz == z}
            parts += [dict(type="plate", w=w, l=ln, x=x, y=y, z=z, color=color)
                      for (x, y, w, ln) in tile_cells(layer, z % 2)]
        # what a voxel cleanup does: stray groups that touch the rest by nothing but a side are left out
        drop = set(loose_parts(parts))
        parts = [p for i, p in enumerate(parts) if i not in drop]
        if not verify(parts, None).passed:
            continue
        out = {i for i, p in enumerate(parts) if p["y"] + p["l"] > oy + by}  # the +y wing, past the block
        return parts, out
    raise ValueError("no connected wing body")


# ---------------------------------------------------------------- tasks

def instruction(level: str, obj: str, site: dict, frame: tuple) -> str:
    """The handoff, in the model's own coordinates (after the random swap/mirror)."""
    swap, mirror, X = frame
    axis, other = ("y", "x") if swap else ("x", "y")
    m = (lambda v: X - 1 - v) if mirror else (lambda v: v)
    outward = ("-" if mirror else "+") + axis
    lo, hi = site["y0"], site["y1"]
    an = "an" if obj[0] in "aeiou" else "a"
    if level == "L1":
        return (f"These are the parts of one wing of {an} {obj}; its trailing edge faces {outward}. Add one flap on "
                "its trailing edge: free a strip of the trailing edge and hinge it so it swings about a horizontal "
                "axis.")
    if level == "L2":
        return (f"These are the parts along the trailing edge of a wing (it faces {outward}; rows {other} = "
                f"{lo}..{hi}). Turn the rear of them into a flap: free it and hinge it about a horizontal axis "
                "along the edge.")
    xh = site["xh"]
    bar, clip = (m(xh - 2), m(xh + 1)) if not mirror else (m(xh - 1), m(xh + 2))
    face_bar, face_clip = ("+" + axis, "-" + axis) if not mirror else ("-" + axis, "+" + axis)
    dims = "w=1, l=2" if swap else "w=2, l=1"
    return (f"These are the parts along the trailing edge of a wing. Make a flap beyond the line {axis} = {m(xh)} "
            f"(rows {other} = {lo}..{hi}): clear the column {axis} = {m(xh)} in those rows and the rows {other} = "
            f"{lo - 1} and {other} = {hi + 1} beyond it, out to the edge, so the strip is free on three sides; keep "
            f"the parts beyond the line as the flap, and hinge it with bar plates ({dims}, facing {face_bar}, at "
            f"{axis} = {bar}) and clip plates ({dims}, facing {face_clip}, at {axis} = {clip}) on top of the sheet "
            f"at z = {site['z']}, in rows {other} = {', '.join(map(str, site['hinge_rows']))}.")


def _view(parts: list[dict], site: dict, level: str, wing_ids: set) -> list[int]:
    """L1: the whole wing plus the block parts at its root; L2/L3: a box around the work plus a ring."""
    if level == "L1":
        y_root = min(parts[i]["y"] for i in wing_ids)
        return sorted(set(wing_ids) | {i for i, p in enumerate(parts) if p["y"] + p["l"] > y_root - 2})
    x_lo = site["xh"] - 6
    return [i for i, p in enumerate(parts) if p["x"] + p["w"] > x_lo and p["y"] < site["y1"] + 4
            and p["y"] + p["l"] > site["y0"] - 3]


def case(seed: int, split: str = "train") -> dict | None:
    rng = random.Random(f"r8-flap-{split}-{seed}")
    parts, wing_ids = wing_body(rng)
    found = sites(parts, sorted(wing_ids), rng=rng, limit=1)
    if not found:
        return None
    site = rng.choice(found)
    level = rng.choice(("L1", "L2", "L2", "L3"))
    view = _view(parts, site, level, wing_ids)
    answer = site_answer(parts, site, color=rng.choice(("dark_gray", "light_gray", "black")))
    # a random orientation: the model is turned and mirrored, and so are the answer and the instruction
    swap, mirror = rng.random() < 0.5, rng.random() < 0.5
    X = max(p["x"] + p["w"] for p in parts) + 1
    full = unframe(parts, swap, mirror, X)
    edits = [{"op": "clear", "box": unframe_box(e["box"], swap, mirror, X)} if e["op"] == "clear"
             else {"op": "add", "part": unframe([e["part"]], swap, mirror, X)[0]} for e in answer]
    a = apply_edits(full, edits, local=view)
    v = check(full, a.parts, a.sources, FLAP, stand=False)
    if not all(v[k][0] for k in v if k in KEEP or k.endswith("_surfaces")):
        return None  # the turned answer must still pass (it always should; checked, never assumed)
    obj = rng.choice(OBJECTS)
    return dict(id=f"flap8_{split}_{seed}", kind="hinge", level=level, object=obj, full=full, view=view,
                instruction=instruction(level, obj, site, (swap, mirror, X)), requirements=FLAP, answer=edits,
                site=site, frame=[swap, mirror, X])


# ---------------------------------------------------------------- mistakes

def mistakes(c: dict, rng: random.Random) -> list[list[dict]]:
    """Habitual wrong answers, each different from the reference: hinge plates inside the top layer (one plate
    down), a channel cut only through the top layers (twice as likely: run 7's own habit on thick wings), a side
    channel missing, the hinge plates a stud off."""
    ans = c["answer"]
    clears = [e for e in ans if e["op"] == "clear"]
    adds = [e for e in ans if e["op"] == "add"]
    inside = clears + [{"op": "add", "part": e["part"] | {"z": e["part"]["z"] - 1}} for e in adds]
    no_side = clears[:1] + clears[2:] + adds if rng.random() < 0.5 else clears[:2] + adds
    # the channel only through the top layers (a thin-sheet habit): the layers under it still hold the strip
    zt = max(e["part"]["z"] for e in adds)
    shallow = [{"op": "clear", "box": e["box"][:4] + [zt - 2, zt]} for e in clears] + adds
    off = copy.deepcopy(ans)
    k = rng.choice([n for n, e in enumerate(off) if e["op"] == "add"])
    off[k]["part"][rng.choice(("x", "y"))] += rng.choice((-1, 1))
    return [inside, shallow, shallow, no_side, off]


def feedback_for(c: dict, bad: list[dict]) -> list[str] | None:
    from .edits import EditError

    try:
        a = apply_edits(c["full"], bad, local=c["view"])
    except EditError as e:
        return [f"answer: {e}"]
    v = check(c["full"], a.parts, a.sources, c["requirements"], stand=False)
    keep = {k: x for k, x in v.items() if k in KEEP or k.endswith("_surfaces")}
    if all(x[0] for x in keep.values()):
        return None  # not actually wrong
    return explain(c["full"], a.parts, a.sources, a.changed, keep, c["view"]) or None


def prompt(c: dict) -> list[dict]:
    return [{"role": "system", "content": RULES},
            {"role": "user", "content": c["instruction"] + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(c['full'][i])}" for k, i in enumerate(c["view"]))}]


def row(c: dict, rng: random.Random, p_mistake: float = 0.5) -> dict:
    chat = prompt(c)
    turns = 1
    if rng.random() < p_mistake:
        options = mistakes(c, rng)
        rng.shuffle(options)
        bad, issues = next(((b, i) for b in options for i in [feedback_for(c, b)] if i), (None, None))
        if issues:
            chat += [{"role": "assistant", "content": json.dumps(bad)},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
            turns = 2
    chat.append({"role": "assistant", "content": json.dumps(c["answer"])})
    return {"messages": chat, "family": "flap_r8", "mode": c["level"], "case": c["id"], "turns": turns}


def _job(args):
    seed, split = args
    try:
        return case(seed, split)
    except Exception as e:  # noqa: BLE001 - an odd body is skipped, and counted
        return {"error": f"{type(e).__name__}: {e}"[:200], "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="training seeds tried")
    ap.add_argument("--test-n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if (args.test / "cases.json").exists():
        raise ValueError("refusing to overwrite a frozen test set")
    with ProcessPoolExecutor(args.workers) as pool:
        train = list(pool.map(_job, [(s, "train") for s in range(args.n)], chunksize=4))
        test = list(pool.map(_job, [(100000 + s, "val") for s in range(args.test_n * 3)], chunksize=4))
    errors = [c for c in train + test if c and "error" in c]
    train = [c for c in train if c and "error" not in c]
    test = [c for c in test if c and "error" not in c][:args.test_n]
    rng = random.Random(8)
    rows = [row(c, rng) for c in train]
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in train))
    args.test.mkdir(parents=True, exist_ok=True)
    (args.test / "cases.json").write_text(json.dumps({"cases": test}))
    print(json.dumps({"train_rows": len(rows), "by_level": {lv: sum(r["mode"] == lv for r in rows)
                                                            for lv in ("L1", "L2", "L3")},
                      "two_turn": sum(r["turns"] == 2 for r in rows), "test": len(test), "errors": len(errors),
                      "error_examples": errors[:3]}, indent=1))


if __name__ == "__main__":
    main()
