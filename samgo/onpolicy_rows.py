"""Correction rows from the model's own mistakes: its wrong answer, the explained errors, then the right answer.

Reads an eval_running_gear --out file collected on training problems (--split train), re-poses each failed
problem, explains what is wrong (running_gear_task.explain) and appends a two-turn row to a training file.

    uv run python -m samgo.onpolicy_rows runs/eval/r1_onpolicy_train.json data/running_gear_r2/train.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np

from .running_gear_task import FEEDBACK, axles_of, explain, pose_open, prompt


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    results = json.loads(src.read_text())["results"]
    rows = []
    for r in results:
        if r["passed"] or "answer" not in r:
            continue
        parts = json.loads(Path(f"runs/bases/{r['base']}.json").read_text())
        target = np.load(f"runs/bases/{r['base']}.npz")["target"]
        e = axles_of(parts)
        problem = pose_open(parts, target, {"front": [e[0]], "rear": [e[-1]], "both": e}[r["task"]])
        if problem is None:
            continue
        bad_ok, issues = explain(problem, r["answer"])
        if bad_ok or not issues:
            continue
        messages = prompt(problem) + [
            {"role": "assistant", "content": json.dumps(r["answer"])},
            {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))},
            {"role": "assistant", "content": json.dumps(problem["answer"])}]
        rows.append({"messages": messages, "base": r["base"], "task": r["task"], "turns": 2, "source": "model"})
    with dst.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"appended {len(rows)} correction rows from the model's own mistakes to {dst}")


if __name__ == "__main__":
    main()
