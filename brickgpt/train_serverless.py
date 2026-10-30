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
import signal
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
    ap.add_argument("--trial", default="32:25,64:25", help="batch:steps phases timed first, on the same model")
    ap.add_argument("--batch", type=int, default=None, help="fixed batch size; skips the trial")
    ap.add_argument("--lr", type=float, default=1e-4, help="for batch 32; scaled by sqrt(batch/32)")
    ap.add_argument("--warmup-rows", type=int, default=800)
    ap.add_argument("--faster", type=float, default=1.5, help="batch 64 must be this much faster to be chosen")
    ap.add_argument("--inflight", type=int, default=2, help="steps queued before waiting (pipelining)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--save-every-rows", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--max-eval-tokens", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # A plain kill (SIGTERM) skips `finally` in Python; turn it into an exit so the session is always deleted.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("FIREWORKS_CLIENT_SOURCE", "fireworks-training-skill/2.3.0")
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import TrainOnWhat, get_text_content
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    from fireworks.training.sdk import FireworksClient

    def promote(checkpoint_contains: str, model_id: str) -> str | None:
        """Turn the newest matching sampler checkpoint into a LoRA model before the session (and it) go away.

        (The cookbook starter's helper only matches its own 'starter-tuned' names.)"""
        fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"],
                             base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
        rows = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
                if r.get("promotable") and checkpoint_contains in (r.get("checkpointName") or r.get("name") or "")]
        if not rows:
            say(event="promote_missing", looking_for=checkpoint_contains)
            return None
        target = max(rows, key=lambda r: r.get("createTime", ""))
        fw.promote_session_checkpoint(name=target["name"], output_model_id=model_id, base_model=BASE_MODEL)
        name = f"accounts/{os.environ.get('FIREWORKS_ACCOUNT_ID', 'vvennela')}/models/{model_id}"
        say(event="promoted", model=name, checkpoint=target["name"])
        return name

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
    total_rows = len(datums) * args.epochs
    say(event="rendered", train=len(datums), skipped=skipped, eval=len(eval_rows), rows=total_rows)

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url(os.environ.get("FIREWORKS_BASE_URL",
                                                                                 "https://api.fireworks.ai")))
    enable_serverless_supervised_409_retry(service)
    done = False
    try:
        # One Glimmer run at a time per account: wait for the slot instead of failing.
        for attempt in range(180):
            try:
                if state.get("adapter_path"):  # warm start from the snapshot saved when the last attempt ended
                    tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank, alpha=32)
                    tc.load_adapter(state["adapter_path"]).result()
                    say(event="resumed", rows=state.get("rows"), session=getattr(service, "training_session_id", None))
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

        # Data position is counted in rows, so the batch size can change mid-run (trial, then the winner).
        order = [i for e in range(args.epochs) for i in random.Random(args.seed + e).sample(range(len(datums)), len(datums))]
        rows_done, step = state.get("rows", 0), state.get("step", 0)
        trial = [(int(b), int(n)) for b, n in (x.split(":") for x in args.trial.split(","))] if args.trial else []
        if args.batch:
            state["batch"] = args.batch
        if state.get("batch"):
            trial = []  # fixed by --batch, or resumed after the choice was made
        plan_iter = iter([b for b, n in trial for _ in range(n)])
        chosen = state.get("batch")
        timing: dict[int, list] = {}
        pending = []  # (step, batch, rows, t_submit, fb_future, opt_future), oldest first
        last_saved = rows_done
        t_start = time.time()

        def lr_for(batch: int, rows: int) -> float:
            progress = rows / max(total_rows, 1)
            warm = min(1.0, (rows + batch) / args.warmup_rows)
            return args.lr * math.sqrt(batch / 32) * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

        def finish_oldest():
            nonlocal last_saved
            st, b, r, t_sub, fb_f, opt_f = pending.pop(0)
            fb, opt = fb_f.result(), opt_f.result()
            m = dict(fb.metrics or {})
            n_tok = m.get("response_tokens") or 1.0
            loss = m.get("loss:sum", 0.0) / n_tok
            dt = time.time() - t_sub
            timing.setdefault(b, []).append((time.time(), m.get("total_tokens:sum", 0.0), loss))
            grad = {k2: round(v, 4) for k2, v in (opt.metrics or {}).items() if k2 in ("grad_norm", "grad_norm_rms")}
            elapsed = time.time() - t_start
            rate_rows = max(r - state.get("rows_at_start", 0), 1) / max(elapsed, 1e-9)
            state["rows"], state["step"] = r, st
            say(event="step", step=st, batch=b, rows=r, of=total_rows, loss=round(loss, 4), latency_s=round(dt, 1),
                eta_h=round((total_rows - r) / rate_rows / 3600, 2), **grad)
            if r - last_saved >= args.save_every_rows:
                # Training-state checkpoints cannot be promoted or sampled, so the checkpoint is a promoted sampler
                # copy: testable mid-run, and the resume point if this process dies without its cleanup. state.json
                # is only rewritten once that copy exists, so rows and weights on disk always match.
                try:
                    tc.save_weights_for_sampler(f"{args.run_id}-s{r}").result()
                    model = promote(f"{args.run_id}-s{r}", f"{args.run_id}-s{r}")
                except Exception as e:  # noqa: BLE001
                    model = None
                    say(event="promote_failed", rows=r, detail=str(e)[:200])
                if model:
                    state.update(step=st, rows=r, adapter_path=model, batch=chosen)
                    state_path.write_text(json.dumps(state))
                    last_saved = r
                    say(event="checkpoint", step=st, rows=r, model=model)
                else:  # retry at the next checkpoint, not every step: a failing save blocks training for minutes
                    last_saved = r
                    say(event="checkpoint_failed", rows=r, next_try=r + args.save_every_rows)

        def tok_rate(b: int) -> float:
            ts = timing.get(b, [])[2:]  # skip the first steps of a phase (queue fill)
            if len(ts) < 2:
                return 0.0
            return sum(t[1] for t in ts[1:]) / max(ts[-1][0] - ts[0][0], 1e-9)

        state["rows_at_start"] = rows_done
        while rows_done < total_rows:
            batch = next(plan_iter, None)
            if batch is None:
                if chosen is None:  # trial over: drain, compare, decide
                    while pending:
                        finish_oldest()
                    r32, r64 = tok_rate(32), tok_rate(64)
                    l32 = sum(t[2] for t in timing.get(32, [])[-10:]) / max(len(timing.get(32, [])[-10:]), 1)
                    l64 = sum(t[2] for t in timing.get(64, [])[-10:]) / max(len(timing.get(64, [])[-10:]), 1)
                    # The 64 phase runs after the 32 phase on a further-trained model, so its loss is not a fair
                    # quality comparison; it only has to show no instability (not above the 32 phase's loss).
                    chosen = 64 if r32 and r64 >= args.faster * r32 and l64 <= l32 * 1.05 else 32
                    state["batch"] = chosen
                    say(event="batch_choice", tok_s_32=round(r32), tok_s_64=round(r64), speedup=round(r64 / max(r32, 1e-9), 2),
                        loss_32_last10=round(l32, 4), loss_64_last10=round(l64, 4), chosen=chosen)
                batch = chosen
            idx = order[rows_done:rows_done + batch]
            lr = lr_for(batch, rows_done)
            fb_f = tc.forward_backward([datums[i] for i in idx], "cross_entropy")
            opt_f = tc.optim_step(sdk.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                                                 grad_clip_norm=1.0), emit_grad_norm_metrics=True)
            step += 1
            rows_done += len(idx)
            pending.append((step, batch, rows_done, time.time(), fb_f, opt_f))
            if len(pending) >= args.inflight:
                finish_oldest()
        while pending:
            finish_oldest()

        state.update(step=step, rows=rows_done)
        state["tuned_eval"] = evaluate("tuned")
        promoted = promote(f"{args.run_id}-tuned", f"{args.run_id}-tuned")
        state["promoted_model"] = promoted
        state_path.write_text(json.dumps(state))
        say(event="done", promoted_model=promoted, base=state["base_eval"], tuned=state["tuned_eval"])
        done = True
    finally:
        # Never leave a session holding the account's single Glimmer slot. After a failure, first export the
        # weights (sampler snapshots outlive the session) so a rerun warm-starts at the same data position.
        if not done and "tc" in locals():
            try:
                rows_now = state.get("rows", 0)
                tc.save_weights_for_sampler(f"{args.run_id}-partial-{rows_now}").result()
                # load_adapter needs a promoted model, not a raw snapshot path, so promote before deleting.
                model = promote(f"{args.run_id}-partial-{rows_now}", f"{args.run_id}-partial-{rows_now}")
                state.update(adapter_path=model)
                state_path.write_text(json.dumps(state))
                say(event="partial_saved", rows=rows_now, model=model)
            except Exception as e:  # noqa: BLE001
                say(event="partial_save_failed", detail=str(e)[:200])
        try:
            service.delete_training_session()
            say(event="session_deleted")
        except Exception as e:  # noqa: BLE001
            say(event="delete_failed", detail=str(e)[:200])
        service.close()
        log.close()


if __name__ == "__main__":
    main()
