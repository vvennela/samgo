"""Verifier feedback at inference time: on a failed build, send the checker's corrections back and retry.

Usage: uv run python -m pilot.feedback_eval --provider baseten --tasks runs/pilot/tasks/test_sub.jsonl --n 4 --rounds 3
Writes runs/pilot/feedback/<name>.jsonl and prints the cumulative pass rate by level after each round,
with the mean tokens spent, so the gain can be weighed against the extra compute.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from math import comb

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from pilot.spark_check import PROVIDERS
from samgo.feedback import describe, retry_message
from samgo.tasks import EXACT, parse
from samgo.verifier import verify

LEVELS = ("1", "2", "5", "10", "20", "all")


async def attempt(client, model, messages, effort):
    extra = {"reasoning_effort": effort} if effort else {}
    try:
        r = await client.chat.completions.create(model=model, messages=messages, max_tokens=16000, **extra)
        return r.choices[0].message.content or "", (r.usage.completion_tokens if r.usage else 0), None
    except Exception as e:  # an API error ends this sample as a failure
        return "", 0, str(e)[:200]


async def solve(client, model, row, rounds, effort, sem):
    target = np.array(row["target"], dtype=np.int32)
    messages = list(row["messages"])
    trace = []
    async with sem:
        for rnd in range(rounds + 1):
            text, tokens, error = await attempt(client, model, messages, effort)
            added = parse(text)
            report = verify(row["given"] + added, target, config=EXACT)
            trace.append({"round": rnd, "passed": report.passed, "codes": sorted(report.codes), "tokens": tokens,
                          "error": error})
            if report.passed or error:
                break
            feedback = describe(report, row["given"], added, target)
            trace[-1]["feedback"] = feedback
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": retry_message(feedback)}]
    passed_at = next((t["round"] for t in trace if t["passed"]), None)
    return {"id": row["id"], "level": row["level"], "passed_at": passed_at, "trace": trace}


def pass_at_k(results: list[dict], k: int) -> float:
    """Control: best-of-k independent first attempts per task (unbiased estimator over the n per task).

    Feedback has to beat this at the same number of attempts; NaN when n < k.
    """
    by_task = defaultdict(list)
    for r in results:
        by_task[r["id"]].append(r["trace"][0]["passed"])
    vals = []
    for first in by_task.values():
        n, c = len(first), sum(first)
        if n < k:
            return float("nan")
        vals.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(vals))


def summarize(results: list[dict], rounds: int) -> list[str]:
    by_level = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(r)
    lines = ["feedback: cumulative pass after round i (i+1 attempts) | control: best-of-(i+1) independent attempts",
             "level  " + "  ".join(f"r{i:<4}" for i in range(rounds + 1)) + " | "
             + "  ".join(f"k={i + 1:<3}" for i in range(rounds + 1)) + "  tokens(r0 -> total)"]
    for level in [lv for lv in LEVELS if lv in by_level] + ["ALL"]:
        rows = results if level == "ALL" else by_level[level]
        rates = [sum(r["passed_at"] is not None and r["passed_at"] <= i for r in rows) / len(rows)
                 for i in range(rounds + 1)]
        t0 = np.mean([r["trace"][0]["tokens"] for r in rows])
        total = np.mean([sum(t["tokens"] for t in r["trace"]) for r in rows])
        control = [pass_at_k(rows, i + 1) for i in range(rounds + 1)]
        lines.append(f"{level:<6} " + "  ".join(f"{x:.2f} " for x in rates) + " | "
                     + "  ".join(f"{x:.2f} " for x in control) + f"  {t0:.0f} -> {total:.0f}")
    errors = sum(any(t["error"] for t in r["trace"]) for r in results)
    lines.append(f"{len(results)} samples, {errors} API errors")
    return lines


async def main(args) -> None:
    load_dotenv()
    base_url, key_env, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    client = AsyncOpenAI(api_key=os.environ.get(key_env, "none"), base_url=args.base_url or base_url, max_retries=10)
    rows = [json.loads(line) for line in open(args.tasks)]
    if args.levels:
        rows = [r for r in rows if r["level"] in args.levels.split(",")]
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(solve(client, model, r, args.rounds, args.effort, sem)
                                     for r in rows for _ in range(args.n)))
    out = Path("runs/pilot/feedback")
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{model.replace('/', '_')}_{args.effort or 'default'}"
    with open(out / f"{name}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    report = "\n".join(summarize(results, args.rounds))
    (out / f"{name}.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="baseten")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="", help="override, e.g. a vLLM server for Glimmer")
    ap.add_argument("--tasks", default="runs/pilot/tasks/test_sub.jsonl")
    ap.add_argument("--levels", default="")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3, help="retries after the first attempt")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--name", default="")
    asyncio.run(main(ap.parse_args()))
