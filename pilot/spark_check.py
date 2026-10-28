"""Reference point: Muse Spark (API, prompting only) on a slice of the frozen test set.

Usage: uv run python -m pilot.spark_check [per_k] [model]
Writes runs/pilot/spark/<model>.jsonl and prints pass rate by k.
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from samgo.tasks import EXACT, parse
from samgo.verifier import verify


async def solve(client, model, row, sem):
    async with sem:
        try:
            r = await client.chat.completions.create(model=model, messages=row["messages"], max_tokens=16000)
            text = r.choices[0].message.content or ""
            usage = r.usage.model_dump() if r.usage else {}
        except Exception as e:  # keep going; an API error is a failed sample
            text, usage = "", {"error": str(e)[:200]}
    added = parse(text)
    report = verify(row["given"] + added, np.array(row["target"], dtype=np.int32), config=EXACT) if added else None
    return {"id": row["id"], "k": row["k"], "passed": bool(report and report.passed),
            "codes": sorted(report.codes) if report else ["no_pieces"], "answer": text[-2000:], "usage": usage}


async def main(per_k: int, model: str) -> None:
    load_dotenv()
    client = AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"], base_url="https://api.meta.ai/v1")
    rows, counts = [], defaultdict(int)
    for line in open("runs/pilot/tasks/test.jsonl"):
        r = json.loads(line)
        if counts[r["k"]] < per_k:
            counts[r["k"]] += 1
            rows.append(r)
    sem = asyncio.Semaphore(8)
    results = await asyncio.gather(*(solve(client, model, r, sem) for r in rows))
    out = Path("runs/pilot/spark")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{model}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    by_k = defaultdict(list)
    for r in results:
        by_k[r["k"]].append(r["passed"])
    tokens = [r["usage"].get("completion_tokens", 0) for r in results if r["usage"]]
    print(model, {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_k.items())},
          "mean completion tokens", sum(tokens) // max(len(tokens), 1),
          "errors", sum("error" in r["usage"] for r in results))
    fails = defaultdict(int)
    for r in results:
        for c in r["codes"]:
            fails[c] += 1
    print("failure codes", dict(fails))


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else "muse-spark-1.3"))
