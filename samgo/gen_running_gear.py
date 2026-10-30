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


def _rows(name: str, bases: str, seed: int) -> list[dict]:
    from .running_gear_task import FEEDBACK, apply, axles_of, check, flawed, pose, prompt

    parts = json.loads((Path(bases) / f"{name}.json").read_text())
    target = np.load(Path(bases) / f"{name}.npz")["target"]
    edges = axles_of(parts)
    if len(edges) < 2:
        return []
    rng = random.Random(f"{seed}-{name}")
    out = []
    for strip, label in (([edges[0]], "front"), ([edges[-1]], "rear"), (edges, "both")):
        problem = pose(parts, target, strip)
        ok, _ = check(apply(problem, problem["answer"]), target)
        if not ok:
            continue
        messages = prompt(problem)
        answer = json.dumps(problem["answer"])
        if rng.random() < FEEDBACK_SHARE:
            bad = flawed(problem["answer"], rng)
            bad_ok, issues = check(apply(problem, bad), target)
            if not bad_ok and issues:
                messages += [{"role": "assistant", "content": json.dumps(bad)},
                             {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues[:12]))}]
        messages.append({"role": "assistant", "content": answer})
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
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in args.bases.glob("*.json") if (args.bases / f"{p.stem}.npz").exists())
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_rows, names, [str(args.bases)] * len(names), [args.seed] * len(names)))
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
