"""Run 11 training data: a rotor that turns about a vertical axis on top of any body - the generic skill behind
anything with blades or a platform spinning on top.

The skill: set a 2x2 turntable (base 3680, turning top 3679 one plate above it) on a top face, raised on a column of
2x2 bricks when the blades would otherwise hit something they pass over, and lay long plates across the turning
half as blades (one plate: two blades; a second plate crossing it: four). The checker (`kinematics.rotors`, axis
vertical) passes it when the blades reach out at least two studs and turn all the way round clear of every part,
with every part held by studs.

Bodies are generic (`flap_data.sheet_body`: blocks with irregular thin structures, plates only); no object words.
Levels: L1 the body only, L2 a spot and the blade size, L3 every part. Half the rows first give a habitual wrong
answer (no column where the blades hit something, the blades on the fixed base, no blades, blades at the turntable's
own level), the checker's feedback for it, then the reference answer.

    python -m samgo.vrotor_data --out data/vrotors_r11 --test runs/audit/r11-vrotors
"""

import argparse
import copy
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .edits import apply_edits
from .flap_data import OBJECTS, sheet_body
from .free_edit_test import FEEDBACK, RULES, check, explain
from .primitives import occupancy

VROTOR = [{"name": "rotor", "kind": "rotor", "axis": "vertical", "count": 1}]
KEEP = ("structure", "connected", "rotor")
BLADES = (("8", 1), ("8", 2), ("6", 1), ("6", 2))  # (length, plates: 1 = two blades, 2 = four)
MAX_MAST = 4


def answer(x: int, y: int, z: int, mast: int, n: int, plates: int, along: str = "x", color: str = "black",
           blade_color: str = "light_gray") -> list[dict]:
    """The reference: turntable at (x..x+1, y..y+1) on the top face whose first free plate is z, on `mast` 2x2
    bricks; blades are 2 x n plates centred on it, the first along `along`, a second across it if plates == 2."""
    edits = [{"op": "add", "part": {"type": "brick", "w": 2, "l": 2, "x": x, "y": y, "z": z + 3 * k,
                                    "color": color}} for k in range(mast)]
    zb = z + 3 * mast
    edits += [{"op": "add", "part": {"type": "turntable_base", "w": 2, "l": 2, "x": x, "y": y, "z": zb,
                                     "color": color}},
              {"op": "add", "part": {"type": "turntable_top", "w": 2, "l": 2, "x": x, "y": y, "z": zb + 1,
                                     "color": color}}]
    off = 1 - n // 2
    for k in range(plates):
        a = along if k == 0 else ("y" if along == "x" else "x")
        w, l, px, py = (n, 2, x + off, y) if a == "x" else (2, n, x, y + off)
        edits.append({"op": "add", "part": {"type": "plate", "w": w, "l": l, "x": px, "y": py, "z": zb + 2 + k,
                                            "color": blade_color}})
    return edits


def _spots(occ: dict, cells) -> list[tuple[int, int, int]]:
    """2x2 spots on a top face: four body cells with nothing above them."""
    out = []
    for (x, y, z) in cells:
        top = [(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)]
        if all((a, b, z) in occ and (a, b, z + 1) not in occ for a, b in top):
            out.append((x, y, z + 1))
    return out


def _passes(parts: list[dict], edits: list[dict]) -> bool:
    try:
        ap = apply_edits(parts, edits)
        v = check(parts, ap.parts, ap.sources, VROTOR, stand=False)
    except Exception:  # noqa: BLE001 - an invalid spot is just skipped
        return False
    return all(v[k][0] for k in KEEP)


def fit(parts: list[dict], x: int, y: int, z: int, n: int, plates: int, along: str) -> int | None:
    """The shortest column (0..MAX_MAST bricks) on which these blades turn clear here, or None."""
    for mast in range(MAX_MAST + 1):
        if _passes(parts, answer(x, y, z, mast, n, plates, along)):
            return mast
    return None


def sites(parts: list[dict], rng: random.Random, n: int, plates: int, along: str, tries: int = 12) -> dict | None:
    occ = occupancy(parts)
    spots = _spots(occ, list(occ))
    rng.shuffle(spots)
    for (x, y, z) in spots[:tries]:
        mast = fit(parts, x, y, z, n, plates, along)
        if mast is not None:
            return {"x": x, "y": y, "z": z, "mast": mast, "n": n, "plates": plates, "along": along}
    return None


def vrotor_site(parts: list[dict], ids: list[int], limit: int = 12) -> dict | None:
    """For the pipeline: the top-face spot of the parts `ids` nearest their middle where the biggest blades turn
    clear on the shortest column (largest blades first)."""
    occ = occupancy(parts)
    mine = occupancy([parts[i] for i in ids])
    spots = [s for s in _spots(occ, list(mine)) if all((s[0] + dx, s[1] + dy, s[2] - 1) in mine
                                                       for dx in (0, 1) for dy in (0, 1))]
    if not spots:
        return None
    cx = sum(c[0] for c in mine) / len(mine)
    cy = sum(c[1] for c in mine) / len(mine)
    spots.sort(key=lambda s: ((s[0] + 1 - cx) ** 2 + (s[1] + 1 - cy) ** 2, -s[2]))
    for n, plates in (("8", 2), ("8", 1), ("6", 2), ("6", 1)):
        for (x, y, z) in spots[:limit]:
            mast = fit(parts, x, y, z, int(n), plates, "x")
            if mast is not None:
                return {"x": x, "y": y, "z": z, "mast": mast, "n": int(n), "plates": plates, "along": "x"}
    return None


def instruction(level: str, obj: str, s: dict) -> str:
    an = "an" if obj[0] in "aeiou" else "a"
    blades = "two" if s["plates"] == 1 else "four"
    if level == "L1":
        return (f"These are the parts of {an} {obj}. Add a rotor on its top: something that turns all the way round "
                "about a vertical axis, with blades reaching out over the top.")
    if level == "L2":
        return (f"These are the parts around one spot on a top face (around x = {s['x']}..{s['x'] + 1}, y = "
                f"{s['y']}..{s['y'] + 1}, first free plate z = {s['z']}). Add a rotor there with {blades} blades "
                f"{s['n'] // 2} studs long: a turntable on the top face, raised on a column of 2x2 bricks if the "
                "blades would hit anything they pass over, and 2-wide plates across its turning half as blades.")
    zb = s["z"] + 3 * s["mast"]
    other = "y" if s["along"] == "x" else "x"
    col = (f"a column of {s['mast']} 2x2 bricks at (x = {s['x']}, y = {s['y']}) from z = {s['z']}, then "
           if s["mast"] else "")
    second = f" and a second across it along {other} at z = {zb + 3}" if s["plates"] == 2 else ""
    return (f"These are the parts around one spot on a top face. Add a rotor: {col}a turntable_base (w=2, l=2) at "
            f"(x = {s['x']}, y = {s['y']}, z = {zb}) and a turntable_top on it at z = {zb + 1}; a 2 x {s['n']} plate "
            f"centred on it along {s['along']} at z = {zb + 2}{second}.")


def _view(parts: list[dict], s: dict, level: str) -> list[int]:
    if level == "L1":
        return list(range(len(parts)))
    r = s["n"] // 2 + 3
    return [i for i, p in enumerate(parts)
            if p["x"] < s["x"] + 2 + r and p["x"] + p.get("w", 1) > s["x"] - r and p["y"] < s["y"] + 2 + r
            and p["y"] + p.get("l", 1) > s["y"] - r and s["z"] - 6 <= p["z"] <= s["z"] + 3 * s["mast"] + 8]


def case(seed: int, split: str = "train") -> dict | None:
    rng = random.Random(f"r11-vrotor-{split}-{seed}")
    parts, _ = sheet_body(rng)
    n, plates = rng.choice(BLADES)
    s = sites(parts, rng, int(n), plates, rng.choice(("x", "y")))
    if s is None:
        return None
    level = rng.choice(("L1", "L2", "L2", "L3"))
    view = _view(parts, s, level)
    edits = answer(s["x"], s["y"], s["z"], s["mast"], s["n"], s["plates"], s["along"],
                   color=rng.choice(("black", "dark_gray", "light_gray")),
                   blade_color=rng.choice(("light_gray", "black", "dark_gray", "red", "yellow")))
    a = apply_edits(parts, edits, local=view)
    v = check(parts, a.parts, a.sources, VROTOR, stand=False)
    if not all(v[k][0] for k in KEEP):
        return None
    obj = rng.choice(OBJECTS)
    return dict(id=f"vrotor11_{split}_{seed}", kind="rotor", level=level, object=obj, full=parts, view=view,
                instruction=instruction(level, obj, s), requirements=VROTOR, answer=edits, site=s)


def mistakes(c: dict) -> list[list[dict]]:
    """Habitual wrong answers: no column where one is needed, the blades on the fixed base (no turning half), no
    blades, the blades at the turning half's own level."""
    ans, s = c["answer"], c["site"]
    out = []
    if s["mast"]:
        low = [e for e in ans if e["part"]["type"] != "brick" or e["part"]["w"] != 2 or e["part"]["l"] != 2]
        low = copy.deepcopy(low)
        for e in low:
            e["part"]["z"] -= 3 * s["mast"]
        out += [low, low]
    no_top = copy.deepcopy([e for e in ans if e["part"]["type"] != "turntable_top"])
    for e in no_top:
        if e["part"]["type"] == "plate":
            e["part"]["z"] -= 1
    out.append(no_top)
    out.append([e for e in ans if e["part"]["type"] != "plate"])
    same = copy.deepcopy(ans)
    for e in same:
        if e["part"]["type"] == "plate":
            e["part"]["z"] -= 1
    out.append(same)
    return out


def feedback_for(c: dict, bad: list[dict]) -> list[str] | None:
    from .edits import EditError

    try:
        a = apply_edits(c["full"], bad, local=c["view"])
    except EditError as e:
        return [f"answer: {e}"]
    v = check(c["full"], a.parts, a.sources, c["requirements"], stand=False)
    keep = {k: x for k, x in v.items() if k in KEEP}
    if all(x[0] for x in keep.values()):
        return None
    return explain(c["full"], a.parts, a.sources, a.changed, keep, c["view"]) or None


def prompt(c: dict) -> list[dict]:
    return [{"role": "system", "content": RULES},
            {"role": "user", "content": c["instruction"] + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(c['full'][i])}" for k, i in enumerate(c["view"]))}]


def row(c: dict, rng: random.Random, p_mistake: float = 0.5) -> dict:
    chat, turns = prompt(c), 1
    if rng.random() < p_mistake:
        options = mistakes(c)
        rng.shuffle(options)
        bad, issues = next(((b, i) for b in options for i in [feedback_for(c, b)] if i), (None, None))
        if issues:
            chat += [{"role": "assistant", "content": json.dumps(bad)},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
            turns = 2
    chat.append({"role": "assistant", "content": json.dumps(c["answer"])})
    return {"messages": chat, "family": "vrotor_r11", "mode": c["level"], "case": c["id"], "turns": turns}


def _job(args):
    seed, split = args
    try:
        c = case(seed, split)
        if c is None:
            return None
        return {"case": c, "row": row(c, random.Random(f"row-{split}-{seed}"))}
    except Exception as e:  # noqa: BLE001 - an odd body is skipped, and counted
        return {"error": f"{type(e).__name__}: {e}"[:200], "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True, help="frozen held-out generated bodies")
    ap.add_argument("--n", type=int, default=2000)
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
    test = [c["case"] for c in test if c and "error" not in c][:args.test_n]
    rows = [c["row"] for c in train]
    random.Random(11).shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "cases.jsonl").write_text("".join(json.dumps(c["case"]) + "\n" for c in train))
    args.test.mkdir(parents=True, exist_ok=True)
    (args.test / "cases.json").write_text(json.dumps({"cases": test}))
    print(json.dumps({"train_rows": len(rows), "by_level": {lv: sum(r["mode"] == lv for r in rows)
                                                            for lv in ("L1", "L2", "L3")},
                      "masts": [sum(c["case"]["site"]["mast"] == m for c in train) for m in range(MAX_MAST + 1)],
                      "two_turn": sum(r["turns"] == 2 for r in rows), "test": len(test), "errors": len(errors),
                      "error_examples": errors[:3]}, indent=1))


if __name__ == "__main__":
    main()
