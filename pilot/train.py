"""One expert-iteration round: LoRA SFT of the base model on passing samples.

Keeps at most --cap passing completions per task (distinct answers first) so
easy tasks don't dominate, trains on completion tokens only (prompt masked),
and logs every optimizer step to <out>/train_log.jsonl for monitoring.
"""

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SKIP = re.compile(r"vision|visual|patch|perception|projector|mm_|image|encoder")


def select(samples_path: Path, cap: int, seed: int) -> list[dict]:
    by_task = defaultdict(list)
    for line in open(samples_path):
        r = json.loads(line)
        if r["passed"]:
            by_task[r["id"]].append(r)
    rng = random.Random(seed)
    chosen = []
    for rows in by_task.values():
        rng.shuffle(rows)
        seen, picked = set(), []
        for r in rows:  # distinct answers first, then repeats if still under the cap
            key = r["completion"].split("to=user<|message|>")[-1]
            if key not in seen:
                seen.add(key)
                picked.append(r)
        picked += [r for r in rows if r not in picked]
        chosen += picked[:cap]
    rng.shuffle(chosen)
    return chosen


def load_model(path: str):
    # On GPUs, spread the BF16 model across all of them; CPU (local smoke tests) stays plain float32.
    # "balanced", not "auto": auto packed ~21 of 22 GB per A10G with weights and OOM'd in backward.
    kwargs = {"dtype": torch.bfloat16, "device_map": "balanced"} if torch.cuda.is_available() else {"dtype": torch.float32}
    try:
        from transformers import AutoModelForImageTextToText as Auto
        return Auto.from_pretrained(path, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForCausalLM as Auto
        return Auto.from_pretrained(path, **kwargs)


def completion_loss(model, ids: torch.Tensor, prompt_len: int, chunk: int = 512) -> torch.Tensor:
    """Mean cross-entropy over completion tokens, computing logits 512 positions at a time.

    The full-vocabulary logits for a ~4k-token sequence (~200k vocab, fp32) need ~3 GB plus as
    much again for their gradient, which ran the last GPU out of memory. Here each chunk's
    logits are rebuilt during backward (checkpoint), so only one chunk is ever held. The head
    matches the model's own: lm_head, then T * tanh(logits * multiplier / T) when softcapped.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config.get_text_config()
    mult = getattr(cfg, "output_multiplier", 1.0) or 1.0
    cap = getattr(cfg, "final_logit_softcapping", None)
    hidden = base.model(input_ids=ids).last_hidden_state[0]  # final-normed hidden states
    # Position t predicts token t+1, so completion tokens [P, L) come from hidden [P-1, L-1).
    h = hidden[prompt_len - 1: ids.shape[1] - 1]
    targets = ids[0, prompt_len:]

    def chunk_loss(h_chunk, t_chunk):
        logits = base.lm_head(h_chunk).float() * mult
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return torch.nn.functional.cross_entropy(logits, t_chunk.to(logits.device), reduction="sum")

    total = 0.0
    for s in range(0, h.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(chunk_loss, h[s:s + chunk], targets[s:s + chunk],
                                                          use_reentrant=False)
    return total / targets.numel()


def lora_target_names(model) -> list[str]:
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in LORA_TARGETS and not SKIP.search(n)]
    if not names:
        raise RuntimeError("no LoRA target modules found in the language model")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=7000, help="skip longer examples (memory)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    data = [d for d in select(args.samples, args.cap, args.seed)
            if len(d["prompt_ids"]) + len(d["completion_ids"]) <= args.max_tokens][: args.max_examples]
    if not data:
        raise SystemExit("no passing samples to train on")

    model = load_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = lora_target_names(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = next(p.device for p in model.parameters() if p.requires_grad)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(data) * args.epochs / args.accum)
    warmup = max(1, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(total_steps - warmup, 1)))

    meta = {"examples": len(data), "tasks": len({d["id"] for d in data}), "lora_modules": len(targets),
            "trainable_params": trainable, "steps": total_steps,
            "examples_by_k": {k: sum(d["k"] == k for d in data) for k in sorted({d["k"] for d in data})}}
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("[train]", json.dumps(meta), flush=True)

    log = open(args.out / "train_log.jsonl", "w")
    model.train()
    started, step, micro, running, running_tokens = time.time(), 0, 0, 0.0, 0
    for epoch in range(args.epochs):
        for ex in data:
            ids = torch.tensor([ex["prompt_ids"] + ex["completion_ids"]], device=device)
            loss = completion_loss(model, ids, len(ex["prompt_ids"]))
            (loss / args.accum).backward()
            running += loss.item()
            running_tokens += len(ex["completion_ids"])
            micro += 1
            if micro % args.accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                row = {"step": step, "of": total_steps, "loss": running / args.accum, "grad_norm": grad_norm,
                       "lr": sched.get_last_lr()[0], "completion_tokens": running_tokens,
                       "elapsed_s": time.time() - started}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print("[train]", json.dumps(row), flush=True)
                running, running_tokens = 0.0, 0
    model.save_pretrained(args.out / "adapter")
    print("[train] saved adapter", flush=True)


if __name__ == "__main__":
    main()
