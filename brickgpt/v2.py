"""SAMgo v2 data: StableText2Brick shapes rebuilt with the 35-part palette, captions kept.

Each structure's occupancy is refined to plate height (20 x 20 x 60) and rebuilt in two passes:
1. structure: the S4 builder tiles it with v2 brick and plate sizes;
   If no layout passes, BrickGPT's own (stable) brick layout of the shape is the base instead.
2. finish: parts are swapped for richer ones only where that cannot break a connection, because the swap
   keeps every stud and grip that was in use: tiles on exposed plate tops, plate + plate + tile caps,
   slopes and curved slopes (on a plate) on exposed edges, inverted slopes over overhangs, arches over
   gaps, round parts on posts.
The result must pass the v2 structure and stability checks to be kept.

    uv run python -m brickgpt.v2 --captions 2 --workers 12
"""

import argparse
import json
import random
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from samgo.builder import _layout
from samgo.palette import KINDS, Part, check, stability

WORLD, LAYERS = 20, 60
SIZES = {k: [s for s in KINDS[k].sizes] for k in ("brick", "plate")}
GRAY = 10  # light gray index for the builder (structures are uncolored, as in BrickGPT)

SYSTEM = "You are a helpful assistant."


def instruction(caption: str) -> str:
    sizes = "; ".join(f"{k} " + ", ".join(f"{a}x{b}" for a, b in v.sizes) for k, v in KINDS.items())
    return ("Create a LEGO model of the input. Format your response as a list of parts, one per line: "
            "<part> <w>x<l> (x,y,z), plus a facing (+x, -x, +y or -y) for slope, invslope and curve.\n"
            "w is the size along x and l along y, in studs; z counts plates (a brick is 3 plates, a curve 2). "
            f"The grid is {WORLD}x{WORLD} studs and {LAYERS} plates high.\n"
            f"Parts: {sizes}. Slope, invslope and curve sizes are width x depth, and the depth runs along "
            "the facing, the direction the slope goes down.\n"
            "Slopes have studs only on their high row; tiles and curves have none; arches rest on their ends.\n\n"
            f"### Input:\n{caption}")


def messages(caption: str, lines: str | None = None) -> list[dict]:
    out = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction(caption)}]
    if lines is not None:
        out.append({"role": "assistant", "content": lines})
    return out


def occupancy(bricks_txt: str) -> np.ndarray:
    occ = np.zeros((WORLD, WORLD, LAYERS), dtype=bool)
    for line in bricks_txt.strip().splitlines():
        h, w, x, y, z = map(int, line.replace("x", " ").replace("(", " ").replace(")", " ").replace(",", " ").split())
        occ[x:x + h, y:y + w, 3 * z:3 * z + 3] = True
    return occ


class Grid:
    """Who occupies each cell, for the 'is anything above / beside / below' questions of the finish pass."""

    def __init__(self, parts: list[Part]):
        self.owner = np.full((WORLD, WORLD, LAYERS), -1, dtype=np.int32)
        for i, p in enumerate(parts):
            self.owner[p.box] = i

    def empty(self, xs: slice, ys: slice, zs: slice) -> bool:
        if xs.start < 0 or ys.start < 0 or xs.stop > WORLD or ys.stop > WORLD or zs.start < 0 or zs.stop > LAYERS:
            return True  # outside the grid counts as open air
        return bool((self.owner[xs, ys, zs] < 0).all())

    def above_empty(self, p: Part) -> bool:
        return p.z + p.h >= LAYERS or self.empty(p.box[0], p.box[1], slice(p.z + p.h, p.z + p.h + 1))

    def side_empty(self, p: Part, facing: str) -> bool:
        """The column of cells just past the edge the slope would face, over the part's height."""
        zs = slice(p.z, p.z + p.h)
        if facing == "+x":
            return self.empty(slice(p.x + p.w, p.x + p.w + 1), p.box[1], zs)
        if facing == "-x":
            return self.empty(slice(p.x - 1, p.x), p.box[1], zs)
        if facing == "+y":
            return self.empty(p.box[0], slice(p.y + p.l, p.y + p.l + 1), zs)
        return self.empty(p.box[0], slice(p.y - 1, p.y), zs)

    def below(self, x: int, y: int, z: int) -> bool:
        return z > 0 and self.owner[x, y, z - 1] >= 0


def _slope_facings(p: Part) -> list[str]:
    """Facings for which this footprint is a valid slope (depth 2 along the facing)."""
    out = []
    if p.l == 2 and (p.w, 2) in KINDS["slope"].sizes:
        out += ["+y", "-y"]
    if p.w == 2 and (p.l, 2) in KINDS["slope"].sizes:
        out += ["+x", "-x"]
    return out


def finish(parts: list[Part], rng: random.Random) -> list[Part]:
    grid = Grid(parts)
    out: list[Part] = []
    for i, p in enumerate(parts):
        open_top = grid.above_empty(p)
        if p.kind == "plate" and open_top and tuple(sorted((p.w, p.l))) in KINDS["tile"].sizes and rng.random() < 0.8:
            out.append(Part("tile", p.w, p.l, p.x, p.y, p.z))
            continue
        if p.kind == "brick":
            facings = [f for f in _slope_facings(p) if grid.side_empty(p, f)]
            if open_top and facings and rng.random() < 0.7:
                f = rng.choice(facings)
                across = p.w if f[1] == "y" else p.l
                if (across, 2) in KINDS["curve"].sizes and rng.random() < 0.35:  # plate + curve, same height
                    out += [Part("plate", p.w, p.l, p.x, p.y, p.z), Part("curve", p.w, p.l, p.x, p.y, p.z + 1, f)]
                else:
                    out.append(Part("slope", p.w, p.l, p.x, p.y, p.z, f))
                continue
            if open_top and tuple(sorted((p.w, p.l))) in KINDS["tile"].sizes and rng.random() < 0.5:  # smooth cap
                out += [Part("plate", p.w, p.l, p.x, p.y, p.z), Part("plate", p.w, p.l, p.x, p.y, p.z + 1),
                        Part("tile", p.w, p.l, p.x, p.y, p.z + 2)]
                continue
            if p.z > 0 and (p.w, p.l) in ((1, 2), (2, 1)):  # inverted slope over an overhang
                cells = [(x, y) for x in range(p.x, p.x + p.w) for y in range(p.y, p.y + p.l)]
                held = [grid.below(x, y, p.z) for x, y in cells]
                if sum(held) == 1 and rng.random() < 0.7:
                    hx, hy = cells[held.index(True)]
                    ox, oy = cells[held.index(False)]
                    f = ("+x" if ox > hx else "-x") if ox != hx else ("+y" if oy > hy else "-y")
                    out.append(Part("invslope", p.w, p.l, p.x, p.y, p.z, f))
                    continue
            if p.z > 0 and tuple(sorted((p.w, p.l))) in ((1, 4), (1, 6)):  # arch over a gap
                cells = ([(x, p.y) for x in range(p.x, p.x + p.w)] if p.w > p.l
                         else [(p.x, y) for y in range(p.y, p.y + p.l)])
                held = [grid.below(x, y, p.z) for x, y in cells]
                if held[0] and held[-1] and not any(held[1:-1]):
                    out.append(Part("arch", p.w, p.l, p.x, p.y, p.z))
                    continue
        if p.kind in ("brick", "plate") and (p.w, p.l) in ((1, 1), (2, 2)):  # round parts on posts
            free = sum(grid.side_empty(p, f) for f in ("+x", "-x", "+y", "-y"))
            if free >= 3 and rng.random() < 0.6:
                out.append(Part("roundbrick" if p.kind == "brick" else "roundplate", p.w, p.l, p.x, p.y, p.z))
                continue
        out.append(p)
    return out


def rebuild(row: dict, seed: int = 0, attempts: int = 4) -> dict | None:
    occ = occupancy(row["bricks"])
    target = np.where(occ, GRAY, 0).astype(np.int32)
    rng = np.random.default_rng(seed)
    for attempt in range(attempts):
        # The builder's layout pass only; validity is judged by the v2 checker (it knows the v2 sizes).
        pieces = _layout(target, None if attempt == 0 else rng, bricks=True, sizes=SIZES)
        base = [Part(p["type"], p["w"], p["l"], p["x"], p["y"], p["z"]) for p in pieces]
        if not check(base, (WORLD, WORLD, LAYERS))["ok"]:
            continue
        parts = finish(base, random.Random(seed * 31 + attempt))
        if not check(parts, (WORLD, WORLD, LAYERS))["ok"]:
            continue
        st = stability(parts)
        if not st["ok"]:
            continue
        return _result(row, parts, st, attempt)
    # Fallback: BrickGPT's own brick layout of this shape (stable by construction), then finished.
    base = [Part("brick", h, w, x, y, 3 * z) for h, w, x, y, z in
            (map(int, re.findall(r"\d+", line)) for line in row["bricks"].strip().splitlines())]
    if check(base, (WORLD, WORLD, LAYERS))["ok"]:
        for k in range(2):
            parts = finish(base, random.Random(seed * 31 + 7 + k))
            if check(parts, (WORLD, WORLD, LAYERS))["ok"] and (st := stability(parts))["ok"]:
                return _result(row, parts, st, "brickgpt_layout")
    return None


def _result(row: dict, parts: list[Part], st: dict, attempt) -> dict:
    parts.sort(key=lambda p: (p.z, p.y, p.x))
    kinds: dict[str, int] = {}
    for p in parts:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    return {"structure_id": row["structure_id"], "lines": "\n".join(p.line() for p in parts), "kinds": kinds,
            "pull": st["peak_pull_per_stud"], "attempt": attempt}


def _work(args):
    row, seed = args
    try:
        return row, rebuild(row, seed)
    except Exception as e:  # one bad structure must not stop the run
        return row, {"error": repr(e)[:200]}


def main() -> None:
    from datasets import load_from_disk

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stabletext2brick/hf")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--captions", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("data/v2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(args.data)[args.split]
    rows = [{"structure_id": r["structure_id"], "bricks": r["bricks"], "captions": r["captions"]}
            for r in (ds.select(range(args.limit)) if args.limit else ds)]
    rng = random.Random(0)
    kept = failed = errors = 0
    totals: dict[str, int] = {}
    with Pool(args.workers) as pool, open(args.out / f"{args.split}_structures.jsonl", "w") as fs, \
            open(args.out / f"{args.split}_c{args.captions}.jsonl", "w") as fc:
        for n, (row, res) in enumerate(pool.imap_unordered(_work, [(r, i) for i, r in enumerate(rows)], chunksize=8), 1):
            if res is None:
                failed += 1
            elif "error" in res:
                errors += 1
            else:
                kept += 1
                fs.write(json.dumps(res) + "\n")
                for k, v in res["kinds"].items():
                    totals[k] = totals.get(k, 0) + v
                for c in rng.sample(row["captions"], min(args.captions, len(row["captions"]))):
                    fc.write(json.dumps({"messages": messages(c, res["lines"])}) + "\n")
            if n % 500 == 0 or n == len(rows):
                print(f"[v2] {n}/{len(rows)} kept {kept} failed {failed} errors {errors} parts {totals}", flush=True)


if __name__ == "__main__":
    main()
