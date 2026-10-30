"""SFT rows for the running-gear task from verified bases (spec.md section 0, training data steps 2-4).

Per base car: three problems (running gear taken off the front axle line, the rear one, both). Each reference
answer is applied and re-checked; rows that do not pass are dropped. About 20% of rows are two-turn: a typical
wrong first answer, the verifiers' real error text, then the correct answer, so the model learns to use feedback.
Vehicles are split by id: 10% go to validation. The photo cars are never used (they are the test set).

    uv run python -m samgo.gen_running_gear --bases runs/bases --out data/running_gear
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FEEDBACK_SHARE = 0.2
CORRECTION_SHARE = 0.4  # open task: wrong answers of the kinds the model really makes, explained, then fixed


def _rows(name: str, bases: str, seed: int, open_task: bool = False) -> list[dict]:
    from .running_gear_task import (FEEDBACK, apply, axles_of, check, explain, flawed, pose, pose_open, prompt,
                                    realistic_mistake)

    parts = json.loads((Path(bases) / f"{name}.json").read_text())
    target = np.load(Path(bases) / f"{name}.npz")["target"]
    edges = axles_of(parts)
    if len(edges) < 2:
        return []
    rng = random.Random(f"{seed}-{name}")
    out = []
    for strip, label in (([edges[0]], "front"), ([edges[-1]], "rear"), (edges, "both")):
        problem = (pose_open if open_task else pose)(parts, target, strip)
        if problem is None:
            continue
        ok, _ = check(apply(problem, problem["answer"]), target)
        if not ok:
            continue
        messages = prompt(problem)
        answer = json.dumps(problem["answer"])
        if open_task and rng.random() < CORRECTION_SHARE:
            bad = realistic_mistake(problem["answer"], rng)
            bad_ok, issues = explain(problem, bad)
            if not bad_ok and issues:
                messages += [{"role": "assistant", "content": json.dumps(bad)},
                             {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
        elif rng.random() < FEEDBACK_SHARE:
            bad = flawed(problem["answer"], rng)
            bad_ok, issues = check(apply(problem, bad), target)
            if not bad_ok and issues:
                messages += [{"role": "assistant", "content": json.dumps(bad)},
                             {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues[:12]))}]
        messages.append({"role": "assistant", "content": answer})
        out.append({"messages": messages, "base": name, "task": label, "turns": (len(messages) - 1) // 2})
    return out


def _frame_rows(name: str, bases: str, seed: int) -> list[dict]:
    """Rows for the frame task (pose_frame): one axle line per answer, as the step-by-step loop asks - the front
    with the rear already done, the rear with the front done, and the front first with the rear still missing.
    CORRECTION_SHARE of rows are two-turn: a wrong answer, the actionable messages (explain), the right answer."""
    from .running_gear_task import FEEDBACK, explain, frame_mistake, frame_problem, prompt

    parts = json.loads((Path(bases) / f"{name}.json").read_text())
    target = np.load(Path(bases) / f"{name}.npz")["target"]
    rng = random.Random(f"{seed}-{name}")
    problems = []
    for label in ("front", "rear", "front_first"):
        pr, partial = frame_problem(parts, target, label)
        if pr is not None:
            problems.append((pr, label, partial))
    out = []
    for pr, label, partial in problems:
        messages = prompt(pr)
        if rng.random() < CORRECTION_SHARE:
            bad = frame_mistake(pr["answer"], rng)
            bad_ok, issues = explain(pr, bad, partial=partial)
            if not bad_ok and issues:
                messages += [{"role": "assistant", "content": json.dumps(bad)},
                             {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
        messages.append({"role": "assistant", "content": json.dumps(pr["answer"])})
        out.append({"messages": messages, "base": name, "task": label, "turns": (len(messages) - 1) // 2})
    return out


def split_of(name: str) -> str:
    uid = name.rsplit("_", 1)[0]
    return "val" if int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 10 == 0 else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", type=Path, default=Path("runs/bases"))
    ap.add_argument("--out", type=Path, default=Path("data/running_gear"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open", action="store_true", help="the open task: no beam positions given (pose_open)")
    ap.add_argument("--frame", action="store_true", help="the frame task on frame-and-shell cars (pose_frame)")
    ap.add_argument("--limit", type=int, default=0, help="first N bases only (0 = all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in args.bases.glob("*.json") if (args.bases / f"{p.stem}.npz").exists())
    names = names[: args.limit or None]
    with ProcessPoolExecutor(args.workers) as pool:
        if args.frame:
            results = list(pool.map(_frame_rows, names, [str(args.bases)] * len(names), [args.seed] * len(names)))
        else:
            results = list(pool.map(_rows, names, [str(args.bases)] * len(names), [args.seed] * len(names),
                                    [args.open] * len(names)))
    files = {s: (args.out / f"{s}.jsonl").open("w") for s in ("train", "val")}
    counts = Counter()
    for name, rows in zip(names, results):
        s = split_of(name)
        for r in rows:
            files[s].write(json.dumps(r) + "\n")
            counts[(s, r["task"], r["turns"])] += 1
    for f in files.values():
        f.close()
    print(f"{len(names)} bases ->", {f"{s}/{t}/{n}-turn": c for (s, t, n), c in sorted(counts.items())})
    print({s: sum(c for (ss, _, _), c in counts.items() if ss == s) for s in ("train", "val")})


if __name__ == "__main__":
    main()
