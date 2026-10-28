"""Reference point: Muse Spark (API, prompting only) on the frozen test set.

Usage: uv run python -m pilot.spark_check [--provider meta|baseten] [--tasks F] [--n N] [--per-k K]
Writes runs/pilot/spark/<model>.jsonl and prints pass rate by level.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from samgo.tasks import EXACT, parse
from samgo.verifier import verify


async def solve(client, model, row, sem, effort):
    async with sem:
        try:
            extra = {"reasoning_effort": effort} if effort else {}
            r = await client.chat.completions.create(model=model, messages=row["messages"], max_tokens=16000, **extra)
            text = r.choices[0].message.content or ""
            usage = r.usage.model_dump() if r.usage else {}
        except Exception as e:  # keep going; an API error is a failed sample
            text, usage = "", {"error": str(e)[:200]}
    added = parse(text)
    report = verify(row["given"] + added, np.array(row["target"], dtype=np.int32), config=EXACT) if added else None
    return {"id": row["id"], "k": row["k"], "level": row.get("level"), "passed": bool(report and report.passed),
            "codes": sorted(report.codes) if report else ["no_pieces"], "answer": text[-2000:], "usage": usage}


PROVIDERS = {  # base URL, key env var, default model
    "meta": ("https://api.meta.ai/v1", "MODEL_API_KEY", "muse-spark-1.3"),
    "baseten": ("https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open-1026"),
}


async def main(args) -> None:
    load_dotenv()
    base_url, key_env, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    client = AsyncOpenAI(api_key=os.environ[key_env], base_url=base_url, max_retries=10)  # backs off on 429
    rows, counts = [], defaultdict(int)
    for line in open(args.tasks):
        r = json.loads(line)
        if not args.per_k or counts[r["k"]] < args.per_k:
            counts[r["k"]] += 1
            rows += [r] * args.n
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(solve(client, model, r, sem, args.effort) for r in rows))
    out = Path("runs/pilot/spark")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{model.replace('/', '_')}{'_' + args.effort if args.effort else ''}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    by_k = defaultdict(list)
    for r in results:
        by_k[r["level"] or r["k"]].append(r["passed"])
    tokens = [r["usage"].get("completion_tokens", 0) for r in results if r["usage"]]
    print(model, {k: f"{sum(v)}/{len(v)}" for k, v in by_k.items()},
          "mean completion tokens", sum(tokens) // max(len(tokens), 1),
          "errors", sum("error" in r["usage"] for r in results))
    fails = defaultdict(int)
    for r in results:
        for c in r["codes"]:
            fails[c] += 1
    print("failure codes", dict(fails))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="meta")
    ap.add_argument("--model", default="")
    ap.add_argument("--tasks", default="runs/pilot/tasks/test.jsonl")
    ap.add_argument("--n", type=int, default=1, help="samples per task")
    ap.add_argument("--per-k", type=int, default=0, help="first K tasks per k (0 = all)")
    ap.add_argument("--effort", default="", help="reasoning_effort (Glimmer pilot used low)")
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main(ap.parse_args()))
