"""Run 10 training data: a rotor on a face of any body - the generic skill behind fans, propellers and turbines.

The skill: set a bearing (a 1x2 Technic brick) into a face of the body - clearing its slot first - run an axle along
the face's normal through its round hole and out of the face, and lock arms on the axle (2-long thick liftarms with
an axle hole) outside the face, with bushes holding the axle. The checker (`kinematics.rotors`) passes it when the
axle and what is locked on it turn all the way round, clear of every part and above the table, with every part held
(the bearing gripped by studs in the face).

Bodies are generic (`flap_data.sheet_body`: blocks with irregular thin structures, plates only); never fans,
engines, propellers or any test object, and no such words. Faces point any of four ways. Levels: L1 the body and
which face, L2 a spot on the face, L3 every part. Half the rows first give a habitual wrong answer (adding the
bearing without clearing its slot, arms loose on round holes, arms against the face, the bearing outside the face),
the checker's precise feedback for it, then the reference answer.

    python -m samgo.rotor_data --out data/rotors_r10 --test runs/audit/r10-rotors
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

ROTOR = [{"name": "rotor", "kind": "rotor", "axis": "horizontal", "count": 1}]
KEEP = ("structure", "connected", "rotor")
FACES = ("+x", "-x", "+y", "-y")


def answer(face: str, F: int, b0: int, z: int, color: str = "dark_gray", arm_color: str = "red") -> list[dict]:
    """The reference rotor on the face `face` whose boundary is at coordinate F (body cells < F for "+", >= F for
    "-"), bearing across b0..b0+1 on the other axis, bottom at plate z."""
    A = face[1]
    plus = face[0] == "+"
    r_in = F - 1 if plus else F
    out = (lambda k: F + k) if plus else (lambda k: F - 1 - k)  # the row k studs out from the face
    start = r_in if plus else F - 4  # the axle spans r_in and four rows out: 5 long
    B = "x" if A == "y" else "y"

    def at(a_val, b_val, **kw):  # a position in (A, B) -> part coordinates
        return {A: a_val, B: b_val, **kw}

    brick = {"type": "technic_brick", "w": 2 if A == "y" else 1, "l": 1 if A == "y" else 2, **at(r_in, b0), "z": z,
             "color": color, "hole": "round"}
    axle = {"type": "axle", "axis": A, **at(start, b0 + 1), "z": z, "length": 5}
    def bush(k):
        return {"type": "bush", "axis": A, **at(out(k), b0 + 1), "z": z}

    def arm(k, b, hole):
        return {"type": "liftarm", "axis": B, **at(out(k), b), "z": z, "length": 2, "axle_hole": hole,
                "color": arm_color}
    box = [r_in, r_in, b0, b0 + 1] if A == "x" else [b0, b0 + 1, r_in, r_in]
    return [{"op": "clear", "box": box + [z, z + 2]},
            {"op": "add", "part": brick}, {"op": "add", "part": axle}, {"op": "add", "part": bush(0)},
            {"op": "add", "part": arm(1, b0 + 1, "start")}, {"op": "add", "part": arm(2, b0, "end")},
            {"op": "add", "part": bush(3)}]


def sites(parts: list[dict], rng: random.Random, limit: int = 1, tries: int = 60) -> list[dict]:
    """Verified rotor sites: a face, a 2-wide, 3-plate-tall spot of body cells on it with nothing in front, high
    enough for the arms to clear the table, where the reference answer passes every check."""
    occ = occupancy(parts)
    found = []
    cells = list(occ)
    rng.shuffle(cells)
    for (x, y, z) in cells[:tries * 20]:
        if len(found) >= limit or tries <= 0:
            break
        if z < 5:
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
        edits = answer(face, F, b, z)
        try:
            ap = apply_edits(parts, edits)
            v = check(parts, ap.parts, ap.sources, ROTOR, stand=False)
        except Exception:  # noqa: BLE001 - an invalid spot is just skipped
            continue
        if all(v[k][0] for k in KEEP):
            found.append({"face": face, "F": F, "b0": b, "z": z})
    return found


def rotor_site(parts: list[dict], ids: list[int], face: str, lo: int, hi: int, limit: int = 30) -> dict | None:
    """The spot on the `face` of the parts `ids` nearest Spark's stretch lo..hi (along the face) where a plain rotor
    passes every check, judged on the neighbourhood of the part (big models stay fast)."""
    from .part_layout import local_parts

    big = False  # whole-model checks are fast enough; a cut-out neighbourhood misjudges what holds on from outside
    keep = local_parts(parts, ids) if big else list(range(len(parts)))
    sub = [parts[i] for i in keep]
    occ = occupancy(sub)
    mine = {c for k in (keep.index(i) for i in ids) for c in occupancy([sub[k]])}
    A, plus = face[1], face[0] == "+"
    mid = (lo + hi) / 2
    cands = []
    for (x, y, z) in mine:
        a, b = (x, y) if A == "x" else (y, x)
        F = a + 1 if plus else a
        front = F if plus else F - 1

        def cell(av, bv, zv):
            return (av, bv, zv) if A == "x" else (bv, av, zv)

        if z < 5 or not all(cell(a, b + db, z + dz) in mine for db in (0, 1) for dz in range(3)):
            continue
        if any(cell(front, b + db, z + dz) in occ for db in (0, 1) for dz in range(3)):
            continue
        zs = sorted(zz for (xx, yy, zz) in mine if (xx, yy) == (x, y))
        cands.append((abs(b + 0.5 - mid), abs(z - (zs[0] + zs[-1]) / 2), F, b, z))
    for _, _, F, b, z in sorted(cands)[:limit]:
        try:
            ap = apply_edits(sub, answer(face, F, b, z))
            v = check(sub, ap.parts, ap.sources, ROTOR, stand=False, crop=big)
        except Exception:  # noqa: BLE001
            continue
        if all(v[k][0] for k in KEEP):
            return {"face": face, "F": F, "b0": b, "z": z}
    return None


def instruction(level: str, obj: str, s: dict) -> str:
    A = s["face"][1]
    B = "x" if A == "y" else "y"
    an = "an" if obj[0] in "aeiou" else "a"
    if level == "L1":
        return (f"These are the parts of {an} {obj}. Add a rotor on its face toward {s['face']}: something that "
                "turns all the way round on a horizontal axle coming out of that face.")
    if level == "L2":
        return (f"These are the parts around one spot on a face toward {s['face']} (around {B} = {s['b0']}.."
                f"{s['b0'] + 1}, z = {s['z']}..{s['z'] + 2}). Add a rotor there: a bearing set into the face, an "
                "axle out of it, and arms locked on the axle that turn all the way round.")
    plus = s["face"][0] == "+"
    F = s["F"]
    r_in = F - 1 if plus else F
    out = (lambda k: F + k) if plus else (lambda k: F - 1 - k)
    start = r_in if plus else F - 4
    dims = "w=2, l=1" if A == "y" else "w=1, l=2"
    return (f"These are the parts around one spot on a face toward {s['face']}. Add a rotor: clear the slot {A} = "
            f"{r_in}, {B} = {s['b0']}..{s['b0'] + 1}, z = {s['z']}..{s['z'] + 2} and set a Technic brick ({dims}, "
            f"round hole) there at z = {s['z']}; an axle along {A} on stud edge {B} = {s['b0'] + 1}, z = {s['z']}, "
            f"from {A} = {start}, length 5; bushes at {A} = {out(0)} and {A} = {out(3)}; liftarms of length 2 along "
            f"{B} at {A} = {out(1)} ({B} = {s['b0'] + 1}, axle_hole start) and {A} = {out(2)} ({B} = {s['b0']}, "
            f"axle_hole end), z = {s['z']}.")


def _view(parts: list[dict], s: dict, level: str) -> list[int]:
    if level == "L1":
        return list(range(len(parts)))
    A = s["face"][1]
    B = "x" if A == "y" else "y"
    W = {"x": "w", "y": "l"}
    return [i for i, p in enumerate(parts)
            if p[A] < s["F"] + 6 and p[A] + p[W[A]] > s["F"] - 6 and p[B] < s["b0"] + 8
            and p[B] + p[W[B]] > s["b0"] - 6 and s["z"] - 5 <= p["z"] <= s["z"] + 7]


def case(seed: int, split: str = "train") -> dict | None:
    rng = random.Random(f"r10-rotor-{split}-{seed}")
    parts, _ = sheet_body(rng)
    found = sites(parts, rng)
    if not found:
        return None
    s = found[0]
    level = rng.choice(("L1", "L2", "L2", "L3"))
    view = _view(parts, s, level)
    edits = answer(s["face"], s["F"], s["b0"], s["z"], color=rng.choice(("dark_gray", "light_gray", "black")),
                   arm_color=rng.choice(("red", "black", "light_gray", "yellow")))
    a = apply_edits(parts, edits, local=view)
    v = check(parts, a.parts, a.sources, ROTOR, stand=False)
    if not all(v[k][0] for k in KEEP):
        return None
    obj = rng.choice(OBJECTS)
    return dict(id=f"rotor10_{split}_{seed}", kind="rotor", level=level, object=obj, full=parts, view=view,
                instruction=instruction(level, obj, s), requirements=ROTOR, answer=edits, site=s)


def mistakes(c: dict, rng: random.Random) -> list[list[dict]]:
    """Habitual wrong answers: the bearing added without clearing its slot (twice as likely: the add-only habit),
    arms on round holes (they spin loose), arms against the face, the bearing outside the face."""
    ans = c["answer"]
    adds = [e for e in ans if e["op"] == "add"]
    loose = copy.deepcopy(ans)
    for e in loose:
        if e["op"] == "add" and e["part"]["type"] == "liftarm":
            e["part"].pop("axle_hole", None)
    s = c["site"]
    A, step = s["face"][1], (-1 if s["face"][0] == "+" else 1)
    close = copy.deepcopy(ans)
    for e in close:
        if e["op"] == "add" and e["part"]["type"] == "liftarm":
            e["part"][A] += 2 * step  # two rows toward the body: into the face
    outside = []
    for e in adds:
        q = copy.deepcopy(e)
        if q["part"]["type"] == "technic_brick":
            q["part"][A] -= step  # the bearing in front of the face instead of in it
        outside.append(q)
    return [adds, adds, loose, close, outside]


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
    return {"messages": chat, "family": "rotor_r10", "mode": c["level"], "case": c["id"], "turns": turns}


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
    rng = random.Random(10)
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
                      "two_turn": sum(r["turns"] == 2 for r in rows), "test": len(test), "errors": len(errors),
                      "error_examples": errors[:3]}, indent=1))


if __name__ == "__main__":
    main()
