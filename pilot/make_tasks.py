"""Freeze the pilot task sets.

Levels: last 1, 2, 5, 10, 20 pieces missing, and "all" (build from empty).
test.jsonl:  40 tasks per level, seeds 100000+. Frozen; never trained on.
train.jsonl: 150 tasks per level, seeds 0+. After the baseline, run.sh trains only on
             levels in the model's 10-90% band and holds out the next harder level.
Tasks whose prompt is very long are skipped so prompts stay under ~4k tokens.
"""

import json
import sys
from pathlib import Path

from samgo.tasks import SYSTEM, make_tasks

LEVELS = ("1", "2", "5", "10", "20", "all")
MAX_PROMPT_CHARS = 9000


def rows(tasks, per_level: int):
    counts = {lv: 0 for lv in LEVELS}
    for t in tasks:
        if counts[t.level] >= per_level:
            continue
        prompt = t.prompt()
        if len(prompt) > MAX_PROMPT_CHARS:
            continue
        counts[t.level] += 1
        yield {
            "id": t.id, "family": t.family, "k": t.k, "level": t.level,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "target": t.target.tolist(), "given": t.given, "reference": t.removed,
        }
        if all(c >= per_level for c in counts.values()):
            return


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, seeds, per_level in (("test", range(100000, 102000), 40), ("train", range(0, 4000), 150)):
        with open(out / f"{name}.jsonl", "w") as f:
            n = 0
            for row in rows(make_tasks(seeds, LEVELS), per_level):
                f.write(json.dumps(row) + "\n")
                n += 1
        print(name, n, "tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot/tasks"))
