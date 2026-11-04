"""Run 12 training data: a blade cross on a face of any body - the real-LEGO version of a rotor out of a face.

The skill: set a bearing (a 1x2 Technic brick) into a face of the body - clearing its slot first - run an axle along
the face's normal through its round hole, a bush on it just outside the face, and lock a blade cross on the axle's
END: a round 2x2 plate with an axle hole (the hub, the axle stops inside it), two 2 x `blade` plates crossed on its
studs and a round 2x2 cap. The axle must not run on through the blade plates (a plate has no hole). The checker
(`kinematics.rotors`) passes it when the blade cross turns all the way round, clear of every part and above the table.

Bodies are generic (`flap_data.sheet_body`), faces point any of four ways, levels L1 (the body and the face), L2 (a
spot), L3 (every part). Half the rows first give a habitual wrong answer (the axle through the blades, the bearing
added without clearing its slot, the cross off the axle's end, the bearing outside the face), the checker's
feedback, then the reference answer. No object words.

    python -m samgo.blade_data --out data/blades_r12 --test runs/audit/r12-blades
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
from .rotor_data import FACES, KEEP, ROTOR, _view


def answer(face: str, F: int, b0: int, z: int, blade: int = 6, color: str = "dark_gray",
           blade_color: str = "light_gray") -> list[dict]:
    """The reference blade cross on the face `face` whose boundary is at F (body cells < F for "+", >= F for "-"),
    bearing across b0..b0+1, bottom at plate z: axle 3 long from inside the bearing to 2 rows out, bush on the first
    row out, the cross's hub on the axle's end."""
    A = face[1]
    plus = face[0] == "+"
    B = "x" if A == "y" else "y"
    r_in = F - 1 if plus else F
    start = r_in if plus else F - 2              # axle: r_in .. r_in + 3 (plus) or F - 2 .. F + 1 (minus)
    end = r_in + 3 if plus else F - 2            # the stud line where the axle ends, outside the face
    bush_at = F if plus else F - 1

    def at(a_val, b_val, **kw):
        return {A: a_val, B: b_val, **kw}

    brick = {"type": "technic_brick", "w": 2 if A == "y" else 1, "l": 1 if A == "y" else 2, **at(r_in, b0), "z": z,
             "color": color, "hole": "round"}
    axle = {"type": "axle", "axis": A, **at(start, b0 + 1), "z": z, "length": 3}
    bush = {"type": "bush", "axis": A, **at(bush_at, b0 + 1), "z": z}
    cross = {"type": "blade_cross", "axis": A, **at(end, b0 + 1), "z": z, "blade": blade, "facing": face[0],
             "color": blade_color}
    box = [r_in, r_in, b0, b0 + 1] if A == "x" else [b0, b0 + 1, r_in, r_in]
    return [{"op": "clear", "box": box + [z, z + 2]}, {"op": "add", "part": brick}, {"op": "add", "part": axle},
            {"op": "add", "part": bush}, {"op": "add", "part": cross}]


def sites(parts: list[dict], rng: random.Random, limit: int = 1, tries: int = 60) -> list[dict]:
    """Verified sites: a face, a 2-wide 3-plate-tall spot of body cells with nothing in front, where the reference
    answer (with the largest blade that turns clear) passes every check."""
    occ = occupancy(parts)
    found = []
    cells = list(occ)
    rng.shuffle(cells)
    for (x, y, z) in cells[:tries * 20]:
        if len(found) >= limit or tries <= 0:
            break
        if z < 7:
            continue
        face = rng.choice(FACES)
        A = face[1]
        a, b = (x, y) if A == "x" else (y, x)
        plus = face[0] == "+"
        F = a + 1 if plus else a
        front = F if plus else F - 1

        def cell(av, bv, zv):
            return (av, bv, zv) if A == "x" else (bv, av, zv)

        slot = [cell(a, b + db, z + dz) for db in (0, 1) for dz in range(3)]
        ahead = [cell(front, b + db, z + dz) for db in (0, 1) for dz in range(3)]
        if not all(c in occ for c in slot) or any(c in occ for c in ahead):
            continue
        tries -= 1
        for blade in rng.sample((8, 6, 4), 3):
            try:
                ap = apply_edits(parts, answer(face, F, b, z, blade))
                v = check(parts, ap.parts, ap.sources, ROTOR, stand=False)
            except Exception:  # noqa: BLE001 - an invalid spot is just skipped
                continue
            if all(v[k][0] for k in KEEP):
                found.append({"face": face, "F": F, "b0": b, "z": z, "blade": blade})
                break
    return found


def instruction(level: str, obj: str, s: dict) -> str:
    A = s["face"][1]
    B = "x" if A == "y" else "y"
    an = "an" if obj[0] in "aeiou" else "a"
    if level == "L1":
        return (f"These are the parts of {an} {obj}. Add a blade cross on its face toward {s['face']}: blades that "
                "turn all the way round on a horizontal axle coming out of that face.")
    if level == "L2":
        return (f"These are the parts around one spot on a face toward {s['face']} (around {B} = {s['b0']}.."
                f"{s['b0'] + 1}, z = {s['z']}..{s['z'] + 2}). Add a blade cross there ({s['blade']}-long blades): a "
                "bearing set into the face, an axle out of it, and the blade cross locked on the axle's end.")
    plus = s["face"][0] == "+"
    F = s["F"]
    r_in = F - 1 if plus else F
    start = r_in if plus else F - 2
    end = r_in + 3 if plus else F - 2
    bush_at = F if plus else F - 1
    dims = "w=2, l=1" if A == "y" else "w=1, l=2"
    return (f"These are the parts around one spot on a face toward {s['face']}. Add a blade cross: clear the slot "
            f"{A} = {r_in}, {B} = {s['b0']}..{s['b0'] + 1}, z = {s['z']}..{s['z'] + 2} and set a Technic brick "
            f"({dims}, round hole) there at z = {s['z']}; an axle along {A} on stud edge {B} = {s['b0'] + 1}, "
            f"z = {s['z']}, from {A} = {start}, length 3; a bush at {A} = {bush_at}; a blade_cross on the axle's end "
            f"at {A} = {end} (facing {s['face'][0]}, blade {s['blade']}), z = {s['z']}.")


def case(seed: int, split: str = "train") -> dict | None:
    rng = random.Random(f"r12-blade-{split}-{seed}")
    parts, _ = sheet_body(rng)
    found = sites(parts, rng)
    if not found:
        return None
    s = found[0]
    level = rng.choice(("L1", "L2", "L2", "L3"))
    view = _view(parts, s, level)
    edits = answer(s["face"], s["F"], s["b0"], s["z"], s["blade"], color=rng.choice(("dark_gray", "light_gray", "black")),
                   blade_color=rng.choice(("light_gray", "dark_gray", "black", "white", "red")))
    a = apply_edits(parts, edits, local=view)
    v = check(parts, a.parts, a.sources, ROTOR, stand=False)
    if not all(v[k][0] for k in KEEP):
        return None
    obj = rng.choice(OBJECTS)
    return dict(id=f"blade12_{split}_{seed}", kind="rotor", level=level, object=obj, full=parts, view=view,
                instruction=instruction(level, obj, s), requirements=ROTOR, answer=edits, site=s)


def mistakes(c: dict, rng: random.Random) -> list[list[dict]]:
    """Habitual wrong answers: the axle run on through the blade plates (twice: the rotor habit of a long axle),
    the bearing added without clearing its slot, the cross one row off the axle's end, the bearing outside the face."""
    ans = c["answer"]
    s = c["site"]
    A, plus = s["face"][1], s["face"][0] == "+"
    adds = [e for e in ans if e["op"] == "add"]
    through = copy.deepcopy(ans)
    for e in through:
        if e["op"] == "add" and e["part"]["type"] == "axle":
            e["part"]["length"] = 5
            if not plus:
                e["part"][A] -= 2
    off = copy.deepcopy(ans)
    for e in off:
        if e["op"] == "add" and e["part"]["type"] == "blade_cross":
            e["part"][A] += 1 if plus else -1
    outside = copy.deepcopy(adds)
    for e in outside:
        if e["part"]["type"] == "technic_brick":
            e["part"][A] += 1 if plus else -1
    return [through, through, adds, off, outside]


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
        options = mistakes(c, rng)
        rng.shuffle(options)
        bad, issues = next(((b, i) for b in options for i in [feedback_for(c, b)] if i), (None, None))
        if issues:
            chat += [{"role": "assistant", "content": json.dumps(bad)},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
            turns = 2
    chat.append({"role": "assistant", "content": json.dumps(c["answer"])})
    return {"messages": chat, "family": "blade_r12", "mode": c["level"], "case": c["id"], "turns": turns}


def _job(args):
    seed, split = args
    try:
        return case(seed, split)
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
    test = [c for c in test if c and "error" not in c][:args.test_n]
    rng = random.Random(12)
    rows = [row(c, rng) for c in train]
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in train))
    args.test.mkdir(parents=True, exist_ok=True)
    (args.test / "cases.json").write_text(json.dumps({"cases": test}))
    print(json.dumps({"train_rows": len(rows), "by_level": {lv: sum(r["mode"] == lv for r in rows)
                                                            for lv in ("L1", "L2", "L3")},
                      "by_face": {f: sum(c["site"]["face"] == f for c in train) for f in FACES},
                      "by_blade": {b: sum(c["site"]["blade"] == b for c in train) for b in (4, 6, 8)},
                      "two_turn": sum(r["turns"] == 2 for r in rows), "test": len(test), "errors": len(errors),
                      "error_examples": errors[:3]}, indent=1))


if __name__ == "__main__":
    main()
