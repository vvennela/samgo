"""Sample K completions per task with vLLM and score each with the verifier.

Glimmer writes reasoning to a `to=self` channel ending in <|eom|>, then the
answer ending in <|eot|>. We stop on <|eot|> or <|end_of_text|> only, never on
<|eom|>, and keep special tokens so the channels can be split.

Writes <out>/samples.jsonl (one row per completion, with token ids for
training) and <out>/progress.json after every chunk, for monitoring.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from samgo.tasks import EXACT, parse
from samgo.verifier import verify


def answer_text(completion: str) -> str:
    """The last message not addressed to=self, or empty if the model never answered.

    The completion continues the generation prompt "<|start|>assistant", so it
    looks like " to=self<|message|>…<|eom|><|start|>assistant to=user<|message|>…<|eot|>".
    """
    answer = ""
    for segment in completion.split("<|start|>assistant"):
        if "<|message|>" not in segment:
            continue
        header, body = segment.split("<|message|>", 1)
        if "to=self" in header:
            continue
        answer = body.split("<|eot|>")[0].split("<|eom|>")[0]
    return answer


def prompt_ids(tok, messages: list[dict], reasoning: str) -> list[int]:
    """Chat-template token ids. Render to text first: in transformers 5, tokenize=True returns
    a dict, not ids. The template already writes the BOS token, so don't add special tokens again."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, reasoning_strength=reasoning)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def score_one(args):
    target, given, completion = args
    answer = answer_text(completion)
    added = parse(answer)
    report = verify(given + added, np.array(target, dtype=np.int32), config=EXACT) if added else None
    return {
        "answered": bool(answer.strip()),
        "pieces": len(added),
        "passed": bool(report and report.passed),
        "codes": sorted(report.codes) if report else ["no_answer" if not answer.strip() else "no_pieces"],
        "iou": report.metrics.get("iou") if report else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora", default="")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--levels", default="", help="comma-separated levels to keep, e.g. 2,5,10")
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.tasks)]
    if args.levels:
        keep = set(args.levels.split(","))
        rows = [r for r in rows if r["level"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    eot = tok.convert_tokens_to_ids("<|eot|>")
    common = dict(model=args.model, tensor_parallel_size=args.tp, max_model_len=12288, generation_config="vllm",
                  gpu_memory_utilization=0.88, enable_lora=bool(args.lora), max_lora_rank=16,
                  enable_prefix_caching=True)
    try:  # text only: don't reserve memory for images or video
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **common)
    except (ValueError, KeyError, TypeError) as e:
        print(f"[sample] retrying without video limit: {e}", flush=True)
        llm = LLM(limit_mm_per_prompt={"image": 0}, **common)
    params = SamplingParams(n=args.n, temperature=1.0, top_p=0.95, top_k=64, max_tokens=args.max_tokens,
                            stop_token_ids=[eot, tok.eos_token_id], ignore_eos=True, skip_special_tokens=False)
    lora = LoRARequest("round1", 1, args.lora) if args.lora else None

    started, done, passed, total = time.time(), 0, 0, 0
    out_tokens = 0
    by_level: dict[str, list[int]] = {}
    with open(args.out / "samples.jsonl", "w") as f, ProcessPoolExecutor() as pool:
        for c in range(0, len(rows), args.chunk):
            chunk = rows[c:c + args.chunk]
            prompts = [prompt_ids(tok, r["messages"], args.reasoning) for r in chunk]
            outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, lora_request=lora)
            jobs, meta = [], []
            for r, p, o in zip(chunk, prompts, outputs):
                for s in o.outputs:
                    ids = list(s.token_ids)
                    if ids and ids[-1] == tok.eos_token_id:
                        ids[-1] = eot  # train on the chat template's turn end
                    elif not ids or ids[-1] != eot:
                        ids = ids + ([] if s.finish_reason == "length" else [eot])
                    text = tok.decode(ids, skip_special_tokens=False)
                    jobs.append((r["target"], r["given"], text))
                    meta.append((r, p, ids, text, s.finish_reason))
            for (r, p, ids, text, finish), sc in zip(meta, pool.map(score_one, jobs, chunksize=8)):
                f.write(json.dumps({"id": r["id"], "k": r["k"], "level": r["level"], "family": r["family"], "prompt_ids": p,
                                    "completion_ids": ids, "completion": text, "finish": finish, **sc}) + "\n")
                total += 1
                passed += sc["passed"]
                out_tokens += len(ids)
                by_level.setdefault(r["level"], []).append(sc["passed"])
            f.flush()
            done += len(chunk)
            elapsed = time.time() - started
            (args.out / "progress.json").write_text(json.dumps({
                "tasks_done": done, "tasks_total": len(rows), "samples": total, "pass_rate": passed / max(total, 1),
                "pass_rate_by_level": {k: sum(v) / len(v) for k, v in by_level.items()},
                "mean_completion_tokens": out_tokens / max(total, 1),
                "tokens_per_s": out_tokens / max(elapsed, 1e-9), "elapsed_s": elapsed,
            }, indent=1))
            print(f"[sample] {done}/{len(rows)} tasks, pass {passed}/{total}, {out_tokens / max(elapsed, 1):.0f} tok/s",
                  flush=True)


if __name__ == "__main__":
    main()
