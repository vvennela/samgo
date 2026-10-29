"""Replication eval: decode N test captions and report BrickGPT's Table-1 metrics.

    uv run python -m brickgpt.evaluate --base-url URL --model MODEL --n 250 [--palette v2] [--workers 8]

Metrics per build: valid (no rejected brick survived: always true by construction), stable (final structure
passes our stability check), bricks, rejections, rollbacks, model calls, seconds, gave_up. Writes one JSONL
row per prompt plus a summary, and the .ldr of every build.
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .decode import Completions, Config, generate

TEST_SEED = 0


def test_prompts(n: int) -> list[dict]:
    """BrickGPT evaluates on 250 test prompts; we take n test structures (fixed seed) and one caption each."""
    from datasets import load_from_disk

    test = load_from_disk("data/stabletext2brick/hf")["test"]
    rng = random.Random(TEST_SEED)
    idx = rng.sample(range(len(test)), n)
    return [{"structure_id": test[i]["structure_id"], "caption": rng.choice(test[i]["captions"])} for i in idx]


def main() -> None:
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("completions", "sampler"), default="completions",
                    help="completions: an OpenAI-compatible server (a Fireworks deployment, vLLM); "
                         "sampler: Fireworks serverless sampling (per-token billing, for spot checks)")
    ap.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--model", default=None, help="model or promoted LoRA; with --backend sampler, omit for the base model")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--palette", choices=("brickgpt", "v2"), default="brickgpt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    load_dotenv()
    tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
    if args.backend == "completions" and not args.model:
        raise SystemExit("--model is required with --backend completions")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ldr").mkdir(exist_ok=True)

    if args.palette == "v2":
        from samgo.palette import parse as parse_line

        from .v2 import Structure2, messages
        make = Structure2
    else:
        from .prompts import messages
        from .structure import Structure as make
        from .structure import parse as parse_line

    def prompt(caption: str) -> str:
        text = tok.apply_chat_template(messages(caption), tokenize=False, add_generation_prompt=True)
        return text[: -len("<|start|>assistant")] + "<|start|>assistant to=user<|message|>"

    shared = None
    if args.backend == "sampler":
        from .fw_sampler import Sampler
        shared = Sampler(args.model)

    def run(i_item):
        i, item = i_item
        llm = shared or Completions(args.base_url, args.model, os.environ.get(args.key_env), session=f"eval-{i}")
        res = generate(llm, prompt(item["caption"]), Config(), structure=make(), parse_line=parse_line)
        calls_total[0] += res.calls
        (args.out / "ldr" / f"{i:03d}.ldr").write_text(res.structure.ldr(item["caption"][:40]))
        return {**item, "bricks": len(res.structure.bricks), "stable": res.stable, "rejections": res.rejections,
                "rollbacks": res.rollbacks, "calls": res.calls, "seconds": round(res.seconds, 1),
                "gave_up": res.gave_up, "lines": res.structure.txt()}

    items = test_prompts(args.n)
    calls_total = [0]
    rows = []
    with ThreadPoolExecutor(args.workers) as pool, open(args.out / "builds.jsonl", "w") as f:
        for row in pool.map(run, enumerate(items)):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[eval] {len(rows)}/{len(items)} stable {sum(r['stable'] for r in rows)} "
                  f"bricks {row['bricks']} rollbacks {row['rollbacks']} {row['seconds']}s", flush=True)
    n = len(rows)
    nonempty = [r for r in rows if r["bricks"]]
    summary = {"n": n, "valid": 1.0, "stable": sum(r["stable"] for r in rows) / n,
               "empty": sum(not r["bricks"] for r in rows) / n,
               "mean_bricks": sum(r["bricks"] for r in nonempty) / max(len(nonempty), 1),
               "median_rollbacks": sorted(r["rollbacks"] for r in rows)[n // 2],
               "gave_up": sum(bool(r["gave_up"]) for r in rows) / n,
               "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
               "model_calls": calls_total[0], "backend": args.backend, "model": args.model or "base",
               "brickgpt_paper": {"valid": 1.0, "stable": 0.988, "median_rollbacks": 2, "median_seconds": 40.8}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
