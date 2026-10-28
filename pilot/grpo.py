"""Online GRPO on the reverse-curriculum tasks: sample, score with the verifier, update, repeat.

Each round draws fresh training prompts (never reused), samples G completions each with the current
adapter, rewards 1 for a verifier pass and 0 otherwise (truncation counts as a fail), and updates on
the groups with mixed outcomes only (all-pass and all-fail groups carry no signal, and all-fail ones
hurt). Advantages are reward minus the group mean, without std or length normalization (Dr. GRPO);
the loss is summed over tokens and divided by a fixed constant. Because prompts are fresh, each
round's pass rate is itself the measurement of the current policy; round 0 is the base model.

    python -m pilot.grpo --model M --tasks train.jsonl --out OUT --roll-gpus 0 --train-gpus 1 --max-minutes 50

Writes OUT/rounds.jsonl (one row per round) and OUT/round<r>/adapter.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LEN_CONST = 8192  # Dr. GRPO: fixed divisor instead of each sequence's own length


def advantages(rewards: list[float]) -> list[float] | None:
    """Reward minus the group mean; None when the group has no contrast."""
    if max(rewards) == min(rewards):
        return None
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def draw(rows: list[dict], levels: list[str], per_level: int, rng: random.Random, used: set) -> list[dict]:
    """Fresh prompts: per_level unseen tasks from each level."""
    picked = []
    for level in levels:
        pool = [r for r in rows if r["level"] == level and r["id"] not in used]
        for r in rng.sample(pool, min(per_level, len(pool))):
            used.add(r["id"])
            picked.append(r)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--levels", default="5,10,20")
    ap.add_argument("--per-level", type=int, default=8, help="prompts per level per round")
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--minibatches", type=int, default=4, help="optimizer steps per round")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--roll-gpus", default="0")
    ap.add_argument("--train-gpus", default="1")
    ap.add_argument("--max-minutes", type=float, default=50)
    ap.add_argument("--max-rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    # The rollout process gets its GPUs first; this process only ever sees the training GPUs.
    ctx = mp.get_context("spawn")
    requests, responses = ctx.Queue(), ctx.Queue()
    from pilot.rollout_worker import serve
    worker = ctx.Process(target=serve, args=(requests, responses, {"gpus": args.roll_gpus, "model": args.model}),
                         daemon=True)
    worker.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpus

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from pilot.sample import prompt_ids, score_one
    from pilot.train import completion_loss, load_model, lora_target_names

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    eot = tok.convert_tokens_to_ids("<|eot|>")
    model = load_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=lora_target_names(model), task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    device = next(p.device for p in params)
    print(f"[grpo] trainer ready after {time.time() - started:.0f}s; waiting for vLLM", flush=True)
    assert responses.get(timeout=1800).get("ready")
    print(f"[grpo] vLLM ready after {time.time() - started:.0f}s", flush=True)

    rows = [json.loads(line) for line in open(args.tasks)]
    levels = args.levels.split(",")
    used: set = set()
    log = open(args.out / "rounds.jsonl", "a")
    adapter, round_minutes = None, 0.0
    with ProcessPoolExecutor() as pool:
        for rnd in range(args.max_rounds):
            elapsed = (time.time() - started) / 60
            if rnd and elapsed + round_minutes > args.max_minutes:
                print(f"[grpo] stopping before round {rnd}: {elapsed:.0f} min used, a round takes {round_minutes:.0f}",
                      flush=True)
                break
            t0 = time.time()
            batch = draw(rows, levels, args.per_level, rng, used)
            prompts = [prompt_ids(tok, r["messages"], "low") for r in batch]
            requests.put({"prompts": prompts, "n": args.group, "max_tokens": args.max_tokens,
                          "stop": [eot, tok.eos_token_id], "lora_path": adapter, "lora_id": rnd})
            outputs = responses.get(timeout=3600)["outputs"]
            t_roll = time.time() - t0

            jobs, meta = [], []
            for r, p, outs in zip(batch, prompts, outputs):
                for ids, finish in outs:
                    if ids and ids[-1] == tok.eos_token_id:
                        ids[-1] = eot
                    jobs.append((r["target"], r["given"], tok.decode(ids, skip_special_tokens=False)))
                    meta.append((r, p, ids, finish))
            scores = list(pool.map(score_one, jobs, chunksize=8))
            groups = defaultdict(list)
            for (r, p, ids, finish), sc in zip(meta, scores):
                groups[r["id"]].append((r, p, ids, float(sc["passed"] and finish != "length")))

            by_level = defaultdict(list)
            train = []
            for items in groups.values():
                rewards = [x[3] for x in items]
                by_level[items[0][0]["level"]] += rewards
                adv = advantages(rewards)
                if adv:
                    train += [(p, ids, a) for (_, p, ids, _), a in zip(items, adv) if len(p) + len(ids) <= 12288]

            # Update: a few minibatch steps over the informative samples; loss = A * sum(NLL) / LEN_CONST.
            t1 = time.time()
            model.train()
            rng.shuffle(train)
            steps, grad_norms = 0, []
            if train:
                size = max(1, -(-len(train) // args.minibatches))
                for s in range(0, len(train), size):
                    mb = train[s:s + size]
                    for p, ids, a in mb:
                        x = torch.tensor([p + ids], device=device)
                        nll_sum = completion_loss(model, x, len(p)) * len(ids)
                        (a * nll_sum / LEN_CONST / len(mb)).backward()
                    grad_norms.append(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    steps += 1
                adapter = str(args.out / f"round{rnd}" / "adapter")
                model.save_pretrained(adapter)
            t_train = time.time() - t1
            round_minutes = (time.time() - t0) / 60

            all_rewards = [x for v in by_level.values() for x in v]
            row = {"round": rnd, "policy": "base" if rnd == 0 else f"after {rnd} update(s)",
                   "pass_rate": sum(all_rewards) / len(all_rewards),
                   "by_level": {lv: round(sum(v) / len(v), 3) for lv, v in sorted(by_level.items())},
                   "prompts": len(groups), "informative_groups": sum(advantages([x[3] for x in g]) is not None
                                                                     for g in groups.values()),
                   "train_samples": len(train), "steps": steps, "grad_norm": grad_norms,
                   "truncated": sum(f == "length" for (_, _, _, f) in meta) / len(meta),
                   "mean_tokens": sum(len(m[2]) for m in meta) / len(meta),
                   "rollout_s": round(t_roll), "train_s": round(t_train), "elapsed_min": round((time.time() - started) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("[grpo] " + json.dumps(row), flush=True)
    requests.put(None)
    print("[grpo] done", flush=True)


if __name__ == "__main__":
    main()
