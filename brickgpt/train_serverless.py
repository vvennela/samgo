"""LoRA SFT of Muse Glimmer 30B on brick data with the Fireworks Serverless Training API.

Follows the cookbook's Muse Glimmer starter (same renderer, loss masking and SDK calls) and adds what a
multi-hour run needs: a resumable checkpoint every --save-every steps, resume from the last one, warmup
and cosine learning-rate decay, and a raw base-vs-tuned eval scored with our brick checkers (the model
answers held-out prompts freely, no decoder help: BrickGPT's untrained-model measurement). The session is
deleted after a successful run so the account's single Glimmer slot frees at once; after a failure it is
kept so the run can resume.

Run from the cookbook root (so `training.*` imports) with this repo on PYTHONPATH:
    caffeinate -dimsu python -u brickgpt/train_serverless.py --run-id brickgpt-v1 --palette brickgpt \
        --train data/sft/train_c2.jsonl --eval data/sft/eval_500.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"
STARTER = Path.home() / ".claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0/training/examples/sft"


def score(text: str, palette: str) -> dict:
    """BrickGPT's validity (every line is a placeable part, in order) and stability, on a free-form answer."""
    if palette == "v2":
        from brickgpt.v2 import Structure2 as Make
        from samgo.palette import parse
    else:
        from brickgpt.structure import Structure as Make
        from brickgpt.structure import parse
    s = Make()
    lines = [line for line in text.strip().splitlines() if line.strip()]
    first_bad = None
    for i, line in enumerate(lines):
        why = s.reject_reason(line)
        if why:
            first_bad = {"index": i, "why": why, "line": line[:60]}
            break
        s.add(parse(line))
    valid = bool(lines) and first_bad is None
    return {"lines": len(lines), "valid": valid, "stable": valid and s.stable(), "first_bad": first_bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--palette", choices=("brickgpt", "v2"), required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--max-eval-tokens", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("FIREWORKS_CLIENT_SOURCE", "fireworks-training-skill/2.3.0")
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import TrainOnWhat, get_text_content
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    sys.path.insert(0, str(STARTER))
    from muse_glimmer_starter import _promote_tuned

    run_dir = REPO / "runs/train" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    log = (run_dir / "metrics.jsonl").open("a")

    def say(**kw):
        kw = {"t": time.strftime("%H:%M:%S"), **kw}
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw), flush=True)

    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")
    rows = [json.loads(line) for line in open(REPO / args.train)]
    datums, skipped = [], 0
    for r in rows:
        model_input, weights = renderer.build_supervised_example(r["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
        tokens = [int(t) for t in model_input.to_ints()]
        w = [float(x) for x in weights.tolist()]
        if len(tokens) > args.max_seq_len or not any(x > 0 for x in w[1:]):
            skipped += 1
            continue
        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(tokens[:-1]),
                                loss_fn_inputs={"target_tokens": tokens[1:], "weights": w[1:]}))
    eval_rows = [json.loads(line) for line in open(REPO / args.eval)][: args.n_eval]
    steps_per_epoch = math.ceil(len(datums) / args.batch_size)
    total = steps_per_epoch * args.epochs
    say(event="rendered", train=len(datums), skipped=skipped, eval=len(eval_rows), steps=total)

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url(os.environ.get("FIREWORKS_BASE_URL",
                                                                                 "https://api.fireworks.ai")))
    enable_serverless_supervised_409_retry(service)
    done = False
    try:
        # One Glimmer run at a time per account: wait for the slot instead of failing.
        for attempt in range(180):
            try:
                if state.get("dcp_path"):
                    tc = service.create_training_client_from_state_with_optimizer(state["dcp_path"])
                    say(event="resumed", step=state["step"], session=getattr(service, "training_session_id", None))
                else:
                    tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank, alpha=32)
                    say(event="session", session=getattr(service, "training_session_id", None))
                break
            except Exception as e:  # noqa: BLE001
                if "RATE_LIMIT" not in str(e) and "429" not in str(e):
                    raise
                if attempt % 5 == 0:
                    say(event="waiting_for_slot", detail=str(e)[:160])
                time.sleep(120)
        else:
            raise SystemExit("no serverless slot after 6 h")

        params = sdk.SamplingParams(max_tokens=args.max_eval_tokens, temperature=0.0,
                                    stop=renderer.get_stop_sequences())

        def evaluate(label: str) -> dict:
            path = tc.save_weights_for_sampler(f"{args.run_id}-{label}").result().path
            sampler = service.create_sampling_client(model_path=path, tokenizer=tok)
            prompts = [renderer.build_generation_prompt(r["messages"][:-1]) for r in eval_rows]
            futures = [sampler.sample(prompt=p, num_samples=1, sampling_params=params) for p in prompts]
            results = []
            with (run_dir / f"predictions_{label}.jsonl").open("w") as f:
                for r, fut in zip(eval_rows, futures):
                    out = list(fut.result().sequences[0].tokens)
                    msg, term = renderer.parse_response(out)
                    text = get_text_content(msg)
                    sc = score(text, args.palette) | {"truncated": not term.is_clean, "tokens": len(out)}
                    results.append(sc)
                    f.write(json.dumps({"caption": r["messages"][1]["content"].split("### Input:\n")[-1][:200],
                                        **sc, "answer": text[:8000]}) + "\n")
            close = getattr(sampler, "close", None)
            if close:
                close()
            n = len(results)
            summary = {"label": label, "n": n, "valid": sum(r["valid"] for r in results) / n,
                       "stable": sum(r["stable"] for r in results) / n,
                       "mean_lines": sum(r["lines"] for r in results) / n,
                       "truncated": sum(r["truncated"] for r in results)}
            say(event="eval", **summary)
            return summary

        if "base_eval" not in state:
            state["base_eval"] = evaluate("base")
            state_path.write_text(json.dumps(state))

        order = list(range(len(datums)))
        start = state.get("step", 0)
        t0, tokens_seen = time.time(), 0
        for step in range(start, total):
            epoch, k = divmod(step, steps_per_epoch)
            if k == 0 or step == start:
                order = list(range(len(datums)))
                random.Random(args.seed + epoch).shuffle(order)
            batch = [datums[i] for i in order[k * args.batch_size:(k + 1) * args.batch_size]]
            progress = step / max(total - 1, 1)
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            fb_f = tc.forward_backward(batch, "cross_entropy")
            opt_f = tc.optim_step(sdk.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.999, eps=1e-8,
                                                 weight_decay=0.0, grad_clip_norm=1.0), emit_grad_norm_metrics=True)
            fb, opt = fb_f.result(), opt_f.result()
            m = dict(fb.metrics or {})
            n_tok = m.get("response_tokens") or 1.0
            tokens_seen += m.get("total_tokens:sum", 0.0)
            grad = {k2: round(v, 4) for k2, v in (opt.metrics or {}).items() if "grad" in k2}
            rate = tokens_seen / max(time.time() - t0, 1e-9)
            eta_h = (total - step - 1) * (time.time() - t0) / max(step + 1 - start, 1) / 3600
            say(event="step", step=step + 1, of=total, loss=round(m.get("loss:sum", 0.0) / n_tok, 4), lr=round(lr, 7),
                tok_s=round(rate), eta_h=round(eta_h, 2), **grad)
            if (step + 1) % args.save_every == 0:
                path = tc.save_state(f"{args.run_id}-step-{step + 1}").result().path
                state.update(step=step + 1, dcp_path=path)
                state_path.write_text(json.dumps(state))
                say(event="checkpoint", step=step + 1)

        state["step"] = total
        state["tuned_eval"] = evaluate("tuned")
        promoted = _promote_tuned(service, os.environ.get("FIREWORKS_ACCOUNT_ID", "vvennela"), args.run_id)
        state["promoted_model"] = promoted
        state_path.write_text(json.dumps(state))
        say(event="done", promoted_model=promoted, base=state["base_eval"], tuned=state["tuned_eval"])
        done = True
    finally:
        if done:  # keep the session after a failure so the run can resume from its checkpoints
            try:
                service.delete_training_session()
                say(event="session_deleted")
            except Exception as e:  # noqa: BLE001
                say(event="delete_failed", detail=str(e)[:200])
        service.close()
        log.close()


if __name__ == "__main__":
    main()
