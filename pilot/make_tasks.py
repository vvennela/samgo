"""Freeze the pilot task sets.

test.jsonl:  200 tasks, k in {1, 2, 5, 10} (50 each), seeds 100000+. Frozen; never trained on.
train.jsonl: training tasks with k in {1, 2, 5}, seeds 0+. k = 10 is held out to test transfer.
Tasks whose prompt is very long are skipped so prompts stay under ~4k tokens.
"""

import json
import sys
from pathlib import Path

from samgo.tasks import SYSTEM, make_tasks

MAX_PROMPT_CHARS = 9000


def rows(tasks, per_k: dict[int, int]):
    counts = {k: 0 for k in per_k}
    for t in tasks:
        if counts.get(t.k, per_k.get(t.k, 0)) >= per_k.get(t.k, 0):
            continue
        prompt = t.prompt()
        if len(prompt) > MAX_PROMPT_CHARS:
            continue
        counts[t.k] += 1
        yield {
            "id": t.id, "family": t.family, "k": t.k,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "target": t.target.tolist(), "given": t.given, "reference": t.removed,
        }
        if all(counts[k] >= per_k[k] for k in per_k):
            return


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "test": (range(100000, 101500), {1: 50, 2: 50, 5: 50, 10: 50}),
        "train": (range(0, 3000), {1: 250, 2: 250, 5: 250}),
    }
    for name, (seeds, per_k) in sets.items():
        tasks = make_tasks(seeds, tuple(per_k))
        with open(out / f"{name}.jsonl", "w") as f:
            n = 0
            for row in rows(tasks, per_k):
                f.write(json.dumps(row) + "\n")
                n += 1
        print(name, n, "tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot/tasks"))
