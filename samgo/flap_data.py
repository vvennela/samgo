"""Run 8 training data: flaps on thin structures that look like real voxelized ones - of any kind.

Run 7's sheets were flat slabs of even thickness lying on a block. Thin structures out of SAM 3D + voxelize are not:
their thickness varies (thicker where they join the body, thinner toward an edge, bumpy), their edges are ragged,
their tops step, they leave the body at any height or lie on top of it and overhang, and things hang under them.
Every layer of a voxel model is tiled with plates, so these bodies are too: a voxel set, each z layer tiled with
real plates, alternate layers crossed.

The bodies are deliberately generic (shelves, lids, ledges, panels, fins, overhangs on blocks) and named after no
particular object type; held-out test objects (the 747, and rockets or satellites later) are never imitated: the
point is a builder that generalizes, not one that has seen the test.

Every task's answer is a verified site from samgo.flap_sites (a plain answer there passes the generic checks), so
the data teaches only generic moves: clear a channel on three sides through the structure's full thickness, put the
bar and clip plates on top of it at the hinge line (one plate above the local top), never inside it.

Rows: the reference answer; or a wrong answer (the habitual mistakes - hinge plates inside the top layer, a channel
cut only through the top layers, a missing side channel, a hinge a stud off), the precise feedback for it, then the
fix. Glimmer's own wrong answers are added by samgo.flap_mistakes.

    python -m samgo.flap_data --out data/flaps_r8 --test runs/audit/r8-sheets
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
OBJECTS = ("object", "machine", "cabinet", "table", "boat", "truck", "robot", "building", "bench", "box", "toy",
           "tool", "appliance", "vehicle", "sculpture")


# ---------------------------------------------------------------- bodies

def sheet_body(rng: random.Random) -> tuple[list[dict], set]:
    """A block (sometimes with a second block on it) and a thin structure of irregular thickness: out of a side at
    any height, or lying on top and overhanging. Canonically the flap edge is +x: a side edge of a structure that
    reaches out along +y, or the far end of one that reaches out along +x. Plates only.
    Returns (parts, ids of the structure's parts outside the block)."""
    from .verifier import verify

    import os

    # SAMGO_BODY_SCALE="lo,hi": bodies lo..hi times larger across (heights by the square root), for large models.
    # Unset: the original sizes and the original random sequence.
    scale = tuple(float(v) for v in os.environ["SAMGO_BODY_SCALE"].split(",")) if os.environ.get(
        "SAMGO_BODY_SCALE") else None
    for _ in range(30):
        cells: set = set()
        bx, by, bz = rng.randint(4, 16), rng.randint(4, 14), rng.randint(4, 14)
        s = rng.uniform(*scale) if scale else 1.0
        sz = s ** 0.5
        bx, by, bz = round(bx * s), round(by * s), round(bz * sz)
        ox, oy = 2, 2 + round(14 * s)
        for x in range(ox, ox + bx):
            for y in range(oy, oy + by):
                for z in range(0, bz):
                    cells.add((x, y, z))
        if rng.random() < 0.3:  # a second, smaller block on top: a less box-like body
            ux, uy = rng.randint(2, bx), rng.randint(2, by)
            sx, sy = ox + rng.randint(0, bx - ux), oy + rng.randint(0, by - uy)
            cells |= {(x, y, z) for x in range(sx, sx + ux) for y in range(sy, sy + uy)
                      for z in range(bz, bz + rng.randint(2, 6))}
        along = rng.choice(("y", "x"))  # the direction it reaches out from the block
        face = by if along == "y" else bx  # the length of the block face it leaves from
        reach = round(rng.randint(4, 14) * s)
        w_root = round(rng.randint(3, max(3, min(face + 4, 14))) * (s if scale else 1))
        w_far = max(2, w_root + rng.randint(-w_root // 2, 2))
        on_top = rng.random() < 0.3
        zmid0 = (bz + 1) if on_top else rng.randint(2, bz)  # on top, or out of the side at any height
        t_root = round(rng.randint(2, 6) * sz)
        t_far = rng.randint(2, max(2, t_root))
        toward_edge = rng.choice((0.0, 0.0, 0.5))  # thinner toward the +x edge as well
        step = rng.choice((0, 0, 3, 5, 8))  # the top steps one plate every `step` rows out (0: level)
        drift = rng.choice((0, 0, 1, 2, 3))  # the structure shifts sideways as it reaches out
        jag = rng.choice((0.0, 0.2, 0.4))
        start = -(rng.randint(2, max(2, (bx if along == "y" else by) - 1)) if on_top else 2)
        v0 = rng.randint(-2, max(-2, face - w_root + 2))
        hang = rng.random() < 0.4
        hang_u = rng.randint(1, max(1, reach - 2))
        sheet = set()
        for u in range(start, reach):  # u: distance out from the block face; v: across
            t = max(u, 0)
            f = t / max(reach - 1, 1)
            width = round(w_root + (w_far - w_root) * f)
            a = v0 + round(drift * f * 2) + (rng.choice((-1, 1)) if rng.random() < jag else 0)
            b = a + width + (rng.choice((-1, 1)) if rng.random() < jag else 0)
            zmid = zmid0 + (t // step if step else 0)
            t_here = t_root + (t_far - t_root) * f
            for v in range(a, b):
                c = (v - a) / max(b - a - 1, 1)  # 0 .. 1 toward the +x side edge (for along == "y")
                thin = toward_edge * max(0.0, c - 0.5) * 2 if along == "y" else 0.0
                th = max(2, round(t_here * (1.0 - thin) + (rng.choice((-1, 0, 0, 1)) if jag else 0)))
                z0 = zmid - th // 2 if not on_top else zmid - 1
                for z in range(max(z0, 1 if u >= 0 else 0), z0 + th):
                    sheet.add((ox + v, oy + by - 1 + u, z) if along == "y" else (ox + bx - 1 + u, oy + v, z))
                if hang and u == hang_u and v in (a + 1, a + 2):  # something hanging under it
                    for x2 in range(-1, rng.randint(2, 4)):
                        for z in range(max(z0 - rng.randint(2, 4), 1), max(z0, 1)):
                            sheet.add((ox + v, oy + by - 1 + u + x2, z) if along == "y"
                                      else (ox + bx - 1 + u + x2, oy + v, z))
        cells |= sheet
        if rng.random() < 0.4:  # a second structure on the opposite face
            if along == "y":
                cells |= {(x, 2 * oy + by - 1 - y, z) for (x, y, z) in sheet}
            else:
                cells |= {(2 * ox + bx - 1 - x, y, z) for (x, y, z) in sheet}
        if min(c[0] for c in cells) < 0 or min(c[1] for c in cells) < 0:
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
        out = ({i for i, p in enumerate(parts) if p["y"] + p["l"] > oy + by} if along == "y" else
               {i for i, p in enumerate(parts) if p["x"] + p["w"] > ox + bx})
        if out:
            return parts, out
    raise ValueError("no connected sheet body")


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
        return (f"These are the parts of one flat structure of {an} {obj}. Add one flap on its edge toward "
                f"{outward}: free a strip of that edge and hinge it so it swings about a horizontal axis.")
    if level == "L2":
        return (f"These are the parts along one edge of a flat structure (the edge toward {outward}, rows {other} = "
                f"{lo}..{hi}). Turn the outer part of it into a flap: free it and hinge it about a horizontal axis "
                "along the edge.")
    xh = site["xh"]
    bar, clip = (m(xh - 2), m(xh + 1)) if not mirror else (m(xh - 1), m(xh + 2))
    face_bar, face_clip = ("+" + axis, "-" + axis) if not mirror else ("-" + axis, "+" + axis)
    dims = "w=1, l=2" if swap else "w=2, l=1"
    return (f"These are the parts along one edge of a flat structure. Make a flap beyond the line {axis} = {m(xh)} "
            f"(rows {other} = {lo}..{hi}): clear the column {axis} = {m(xh)} in those rows and the rows {other} = "
            f"{lo - 1} and {other} = {hi + 1} beyond it, out to the edge, so the strip is free on three sides; keep "
            f"the parts beyond the line as the flap, and hinge it with bar plates ({dims}, facing {face_bar}, at "
            f"{axis} = {bar}) and clip plates ({dims}, facing {face_clip}, at {axis} = {clip}) on top of the sheet "
            f"at z = {site['z']}, in rows {other} = {', '.join(map(str, site['hinge_rows']))}.")


def _view(parts: list[dict], site: dict, level: str, ids: set) -> list[int]:
    """L1: the whole structure plus the body parts it touches; L2/L3: a box around the work plus a ring."""
    if level == "L1":
        x0 = min(parts[i]["x"] for i in ids) - 2
        x1 = max(parts[i]["x"] + parts[i]["w"] for i in ids) + 2
        y0 = min(parts[i]["y"] for i in ids) - 2
        y1 = max(parts[i]["y"] + parts[i]["l"] for i in ids) + 2
        return [i for i, p in enumerate(parts) if p["x"] < x1 and p["x"] + p["w"] > x0 and p["y"] < y1
                and p["y"] + p["l"] > y0]
    x_lo = site["xh"] - 6
    return [i for i, p in enumerate(parts) if p["x"] + p["w"] > x_lo and p["y"] < site["y1"] + 4
            and p["y"] + p["l"] > site["y0"] - 3]


def case(seed: int, split: str = "train") -> dict | None:
    rng = random.Random(f"r8-flap-{split}-{seed}")
    parts, ids = sheet_body(rng)
    found = sites(parts, sorted(ids), rng=rng, limit=1)
    if not found:
        return None
    site = rng.choice(found)
    level = rng.choice(("L1", "L2", "L2", "L3"))
    view = _view(parts, site, level, ids)
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
    down), a channel cut only through the top layers (twice as likely: run 7's habit on thick structures), a side
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
    ap.add_argument("--test", type=Path, required=True, help="frozen held-out generated structures")
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
