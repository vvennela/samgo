"""Run 4 data: randomized assembly/repair tasks across object families, with a frozen held-out test set.

Every case uses the audit harness's own task definition (audit_objects: gear / repair / add modes, prompts,
native checks), so training rows and test cases are posed and checked identically. Variety comes from builder
sizes, translating the whole model across the grid, mirroring colors, and which mechanism part is broken and how.

Generalization is measured two ways, decided before any data is generated:
- held-out FAMILIES (never in training): HELD_OUT;
- held-out INSTANCES of the training families: distinct seeds, checked to share no input with any training row.

    uv run python -m samgo.gen_objects --out data/objects_r4 --test runs/audit/r4-frozen
"""

import argparse
import copy
import json
import random

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .assembly import digest, save
from .audit_objects import (
    bed_reference,
    gear_case,
    grid,
    inspect_case,
    messages,
    moving_groups,
    native_checks,
    rolling_reference,
)
from .running_gear_task import FEEDBACK

HELD_OUT = ("skateboard", "microwave")          # whole families kept out of training
TRAIN_FAMILIES = ("car", "train", "chair", "bicycle", "door", "door_var", "tv", "laptop")
CORRECTION_SHARE = 0.4
PALETTE = ("red", "blue", "yellow", "black", "white", "light_gray", "dark_gray", "green", "orange", "tan")


_NAMES: list[str] = []


def _base_names() -> list[str]:
    if not _NAMES:
        _NAMES.extend(sorted(p.stem for p in Path("runs/bases_shell").glob("*.json")
                             if p.with_suffix(".npz").exists()))
    return _NAMES


def shifted(parts: list[dict], dx: int, dy: int) -> list[dict]:
    out = copy.deepcopy(parts)
    for p in out:
        p["x"] += dx
        p["y"] += dy
        if "pin_x" in p:
            p["pin_x"] += dx
        if "pin_y" in p:
            p["pin_y"] += dy
    return out


def recolored(parts: list[dict], rng: random.Random) -> list[dict]:
    """Body colors swapped for others (functional parts keep theirs), so color is no cue to what to fix."""
    body = sorted({p.get("color") for p in parts if p["type"] in ("plate", "brick") and p.get("color")})
    new = dict(zip(body, rng.sample(PALETTE, len(body)))) if len(body) <= len(PALETTE) else {}
    return [p | {"color": new.get(p.get("color"), p.get("color"))} if p["type"] in ("plate", "brick") else p
            for p in parts]


def _model(family: str, rng: random.Random, split: str | None = None) -> tuple[list[dict], dict]:
    """A verified model of `family` with the context its native checks need."""
    if family == "chair":
        from .chair import build_chair

        r = build_chair(size=rng.choice((12, 16)))
        parts = r["parts"]
        i = max(i for i, p in enumerate(parts) if p["type"] == "turntable_top")
        return parts, {"requirements": r["requirements"], "seat": list(range(i, len(parts)))}
    if family == "bicycle":
        from .bicycle import build_bicycle

        r = build_bicycle(length=rng.randint(14, 20))
        return r["parts"], {"requirements": r["requirements"]}
    if family == "skateboard":
        from .skateboard import build_skateboard

        r = build_skateboard(length=rng.choice((16, 20, 24, 30)), width=rng.choice((4, 6)))
        return r["parts"], {"requirements": r["requirements"]}
    if family == "microwave":
        from .doors import build_microwave

        r = build_microwave(size=rng.choice((10, 12)))
        parts = r["parts"]
        doors = moving_groups(parts, {"hinge_top"})
        dish = moving_groups(parts, {"turntable_top"})[0]
        return parts, {"requirements": r["requirements"],
                       "moving": {f"door_{i}": (g, 90) for i, g in enumerate(doors)} | {"dish": (dish, 360)}}
    if family == "door":
        from .doors import CAR_REQUIREMENTS

        name = rng.choice(("camry", "beetle"))
        parts = json.loads(Path(f"runs/demo/doors/{name}_doors.json").read_text())
        groups = moving_groups(parts, {"hinge_top"})
        return parts, {"requirements": CAR_REQUIREMENTS, "vehicle": True,
                       "moving": {f"door_{i}": (g, 90) for i, g in enumerate(groups)}}
    if family == "door_var":
        from .doors import CAR_REQUIREMENTS, add_car_doors
        from .gen_running_gear import split_of

        names = [n for n in _base_names() if split is None or split_of(n) == split]  # test cars: unseen vehicles
        for _ in range(12):
            name = rng.choice(names)
            parts = json.loads(Path(f"runs/bases_shell/{name}.json").read_text())
            target = np.load(f"runs/bases_shell/{name}.npz")["target"]
            try:
                r = add_car_doors(parts, target)
            except ValueError:
                continue
            if all(v if isinstance(v, bool) else v[0] for v in r["checks"].values()):
                groups = moving_groups(r["parts"], {"hinge_top"})
                return r["parts"], {"requirements": CAR_REQUIREMENTS, "vehicle": True, "base": name,
                                    "moving": {f"door_{i}": (g, 90) for i, g in enumerate(groups)}}
        raise ValueError("no door car found")
    if family in ("tv", "laptop"):
        w = rng.choice((4, 6, 8))
        if family == "tv":
            c = rng.randint(0, w - 2)
            parts = [dict(type="plate", w=w, l=2, x=0, y=0, z=0, color="black"),
                     dict(type="turntable_base", w=2, l=2, x=c, y=0, z=1, color="black"),
                     dict(type="turntable_top", w=2, l=2, x=c, y=0, z=2, color="black"),
                     dict(type="brick", w=w, l=1, x=0, y=0, z=3, color="blue")]
            kind, axis = "turntable_top", "vertical"
        else:
            parts = [dict(type="plate", w=w, l=2, x=0, y=0, z=0, color="red"),
                     dict(type="plate", w=w, l=2, x=0, y=2, z=0, color="red"),
                     dict(type="plate", w=w, l=2, x=0, y=1, z=1, color="red"),
                     dict(type="click_a", w=w, l=1, x=0, y=3, z=1, pin_x=0, pin_y=4, pin_axis="x", color="black"),
                     dict(type="click_b", w=w, l=1, x=0, y=4, z=1, pin_x=0, pin_y=4, pin_axis="x", color="black"),
                     dict(type="brick", w=w, l=1, x=0, y=4, z=2, color="blue")]
            kind, axis = "click_b", "horizontal"
        return parts, {"requirements": [dict(name="screen_moves", kind="swing", axis=axis, degrees=60)],
                       "moving": {"screen": (moving_groups(parts, {kind})[0], 60)}}
    raise ValueError(family)


# What can be broken in each family, as (part types, field, deltas): the mechanism-critical placements.
BREAKS = {
    "chair": [(("wheel", "axle"), "z", (-1, 1)), (("turntable_top",), "x", (-1, 1)), (("turntable_top",), "y", (-1, 1)),
              (("half_bush",), "y", (-1, 1))],
    "bicycle": [(("wheel",), "z", (-1, 1)), (("axle",), "z", (-1, 1)), (("turntable_top",), "x", (-1, 1)),
                (("liftarm",), "axle_hole", None)],
    "skateboard": [(("wheel",), "z", (-1, 1)), (("axle",), "z", (-1, 1)), (("technic_brick",), "z", (-1, 1))],
    "microwave": [(("hinge_top",), "pin_x", (-1, 1)), (("turntable_top",), "x", (-1, 1)),
                  (("turntable_top",), "y", (-1, 1))],
    "door": [(("hinge_top",), "pin_x", (-1, 1))],
    "door_var": [(("hinge_top",), "pin_x", (-1, 1))],
    "tv": [(("turntable_top",), "x", (-1, 1)), (("turntable_top",), "z", (1,))],
    "laptop": [(("click_b",), "pin_y", (-1, 1)), (("click_a",), "pin_y", (-1, 1))],
}


def _ok(checks: dict) -> bool:
    return all(v if isinstance(v, bool) else v[0] for v in checks.values())


def repair_variant(family: str, seed: int) -> dict | None:
    """A broken copy of a verified, translated, recolored model: one mechanism placement changed (both halves of
    a hinge when they share a pin). None if the reference fails or the break goes undetected."""
    rng = random.Random(f"{family}-{seed}")
    parts, ctx = _model(family, rng)
    dx, dy = (rng.randint(0, 6), rng.randint(0, 6)) if family not in ("door",) else (0, 0)
    parts = recolored(shifted(parts, dx, dy), rng)
    if "moving" in ctx:  # moving groups are index lists: unchanged by translation
        ctx = copy.deepcopy(ctx)
    case = dict(id=f"{family}_{seed}", family=family, mode="repair", input=parts, reference=parts, context=ctx)
    if not _ok(native_checks(case, parts)):
        return None
    types, field, deltas = rng.choice(BREAKS[family])
    candidates = [i for i, p in enumerate(parts) if p["type"] in types and field in p]
    if not candidates:
        return None
    i = rng.choice(candidates)
    if field == "axle_hole":
        change = {"axle_hole": "end" if parts[i]["axle_hole"] == "start" else "start"}
    else:
        change = {field: parts[i][field] + rng.choice(deltas)}
    group = [i]
    if field.startswith("pin") and parts[i]["type"].startswith("hinge"):  # a door hangs on two hinges
        group = [k for k, p in enumerate(parts) if p["type"] == parts[i]["type"] and p.get("pin_x") == parts[i]["pin_x"]
                 and p.get("pin_y") == parts[i]["pin_y"]]
    broken = copy.deepcopy(parts)
    for k in group:
        broken[k].update(change if field == "axle_hole" else {field: broken[k][field] + (change[field] - parts[i][field])})
    case |= dict(input=broken, oracle=[{"op": "set", "index": k, "part": parts[k]} for k in group])
    if inspect_case(case, []).get("passed"):
        return None  # the break is not caught by the checks: useless as a task
    return case


MECHANISM = {"wheel", "axle", "bush", "half_bush", "turntable_base", "turntable_top", "hinge_base", "hinge_top",
             "click_a", "click_b", "car_door"}


def _is_mechanism(p: dict) -> bool:
    return p["type"] in MECHANISM or (p["type"] == "liftarm" and "axle_hole" in p)  # crank arms


def place_case(parts: list[dict], family: str, ctx: dict, case_id: str, only: set | None = None) -> dict:
    """The placement task for a verified model: its moving-part connections removed (the answer adds them back);
    payload checks name the input parts that stay, so the answer's order is free. `only`: remove just these types
    (e.g. a car's hinges, when its running gear is already in place)."""
    removed = [i for i, p in enumerate(parts) if (p["type"] in only if only else _is_mechanism(p))]
    kept = [i for i in range(len(parts)) if i not in removed]
    new_index = {old: k for k, old in enumerate(kept)} | {old: len(kept) + j for j, old in enumerate(removed)}
    ctx = copy.deepcopy(ctx)
    if "seat" in ctx:
        ctx["seat"] = sorted(new_index[i] for i in ctx["seat"] if i in new_index and i not in removed)
    if "moving" in ctx:
        ctx["moving"] = {k: (sorted(new_index[i] for i in g if i not in removed), d) for k, (g, d) in ctx["moving"].items()}
    return dict(id=case_id, family=family, mode="place", input=[parts[i] for i in kept],
                reference=[parts[i] for i in kept + removed], context=ctx,
                oracle=[{"op": "add", "part": parts[i]} for i in removed])


def place_variant(family: str, seed: int) -> dict | None:
    """A randomized placement task (translated, recolored); None if the reference fails or nothing is missing."""
    rng = random.Random(f"place-{family}-{seed}")
    parts, ctx = _model(family, rng, "val" if seed >= 100000 else "train")
    dx, dy = (rng.randint(0, 6), rng.randint(0, 6)) if family not in ("door", "door_var") else (0, 0)
    only = {"hinge_base", "hinge_top"} if family == "door_var" and rng.random() < 0.7 else None
    case = place_case(recolored(shifted(parts, dx, dy), rng), "door" if family == "door_var" else family, ctx,
                      f"place_{family}_{seed}", only)
    if not _ok(native_checks(case, case["reference"])):
        return None
    if inspect_case(case, []).get("passed"):
        return None  # nothing missing that the checks would notice
    return case


def gear_variant(family: str, seed: int) -> dict | None:
    rng = random.Random(f"{family}-{seed}")
    if family == "car":
        parts, edges = rolling_reference(rng.choice((4, 6, 8)), offset=rng.randint(0, 8))
    else:
        parts, edges = rolling_reference(6, axles=rng.randint(3, 5), offset=rng.randint(0, 6), train=True)
    parts = recolored(parts, rng)
    try:
        return gear_case(f"{family}_{seed}", parts, edges, family)
    except ValueError:
        return None


def variant(family: str, seed: int, mode: str = "repair") -> dict | None:
    if family in ("car", "train"):
        return gear_variant(family, seed)
    return place_variant(family, seed) if mode == "place" else repair_variant(family, seed)


def wrong_answer(case: dict, rng: random.Random) -> list[dict]:
    """A plausible wrong answer: the right part with a different wrong value, or (gear) one placement off."""
    bad = copy.deepcopy(case["oracle"])
    if case["mode"] == "place" and len(bad) > 1 and rng.random() < 0.3:
        bad.remove(rng.choice(bad))  # a connection left out
        return bad
    e = rng.choice(bad)
    field = rng.choice([k for k in ("x", "y", "z") if k in e["part"]])
    e["part"][field] += rng.choice((-1, 1))
    return bad


def row(case: dict, seed: int) -> dict | None:
    rng = random.Random(f"row-{seed}-{case['id']}")
    chat = messages(case)
    if rng.random() < CORRECTION_SHARE:
        bad = wrong_answer(case, rng)
        verdict = inspect_case(case, bad)
        if not verdict["passed"] and verdict["issues"]:
            chat += [{"role": "assistant", "content": json.dumps(bad)},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in verdict["issues"][:12]))}]
    chat.append({"role": "assistant", "content": json.dumps(case["oracle"])})
    return {"messages": chat, "family": case["family"], "mode": case["mode"], "case": case["id"],
            "turns": (len(chat) - 1) // 2, "input_digest": digest(case["input"])}


def _job(args: tuple) -> tuple[dict | None, dict | None]:
    family, seed, make_row, mode = args
    try:
        case = variant(family, seed, mode)
    except Exception as e:  # noqa: BLE001 - a builder variant that cannot be made is skipped, and counted
        return None, {"family": family, "seed": seed, "error": f"{type(e).__name__}: {e}"}
    if case is None:
        return None, {"family": family, "seed": seed, "error": "reference failed or break undetected"}
    return (row(case, seed) if make_row else case), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/objects_r4"))
    ap.add_argument("--test", type=Path, default=Path("runs/audit/r4-frozen"))
    ap.add_argument("--per-family", type=int, default=400, help="training seeds tried per training family")
    ap.add_argument("--test-instances", type=int, default=6, help="held-out instances per training family")
    ap.add_argument("--test-held-out", type=int, default=12, help="cases per held-out family")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", nargs="+", help="only these training families (no held-out families)")
    args = ap.parse_args()
    if (args.test / "cases.json").exists():
        raise ValueError("refusing to overwrite a frozen test set; use a new --test directory")
    # Test seeds are drawn from a separate range so no test instance reuses a training seed.
    modes = lambda f: ("gear",) if f in ("car", "train") else ("place",) if f == "door_var" else ("repair", "place")
    families = tuple(args.families) if args.families else TRAIN_FAMILIES
    held = () if args.families else HELD_OUT
    test_jobs = [(f, 100000 + s, False, m) for f in families for m in modes(f)
                 for s in range(args.test_instances * 3)]
    test_jobs += [(f, 200000 + s, False, m) for f in held for m in modes(f) for s in range(args.test_held_out * 3)]
    count = lambda f: args.per_family // 2 if f in ("door", "door_var") else args.per_family  # big cars check slowly
    train_jobs = [(f, s, True, m) for f in families for m in modes(f) for s in range(count(f))]
    with ProcessPoolExecutor(args.workers) as pool:
        test_out = list(pool.map(_job, test_jobs))
        train_out = list(pool.map(_job, train_jobs))
    cases, per = [], {}
    for (case, _), (family, _, _, mode) in zip(test_out, test_jobs):
        want = args.test_held_out if family in held else args.test_instances
        key = f"{family}/{mode}"
        if case is not None and per.get(key, 0) < want:
            cases.append(case)
            per[key] = per.get(key, 0) + 1
    rows = [r for r, _ in train_out if r is not None]
    test_inputs = {digest(c["input"]) for c in cases}
    leaked = [r for r in rows if r["input_digest"] in test_inputs]
    rows = [r for r in rows if r["input_digest"] not in test_inputs]
    args.test.mkdir(parents=True, exist_ok=True)
    save(args.test / "cases.json", {"cases": cases, "sha256": digest(cases), "held_out_families": list(HELD_OUT),
                                    "train_families": list(TRAIN_FAMILIES)})
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rng.shuffle(rows)
    with (args.out / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    skipped = [e for _, e in train_out + test_out if e]
    print(json.dumps({"test_cases": per, "train_rows": len(rows), "dropped_as_test_duplicates": len(leaked),
                      "train_rows_by_family_mode": {f"{f}/{m}": sum(r["case"].startswith(f"place_{f}_") or (r["family"] == f and r["mode"] == m) for r in rows)
                                                    for f in families for m in ("gear", "repair", "place")},
                      "two_turn_rows": sum(r["turns"] == 2 for r in rows), "skipped_variants": len(skipped),
                      "skip_examples": skipped[:4]}, indent=1))


if __name__ == "__main__":
    main()
