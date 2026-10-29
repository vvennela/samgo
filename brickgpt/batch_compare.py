"""Fair batch-size comparison: each config starts from a fresh LoRA, trains on the same rows in the same
order, and is scored on the same held-out rows (forward-only loss). Sessions run one after another (the
account allows one Glimmer run) and are always deleted.

    python -u brickgpt/batch_compare.py --rows 1600 --configs 32,64
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/sft/train_c2.jsonl"))
    ap.add_argument("--heldout", type=Path, default=Path("data/sft/eval_500.jsonl"))
    ap.add_argument("--rows", type=int, default=1600)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--configs", default="32,64")
    ap.add_argument("--lr", type=float, default=1e-4, help="for batch 32; scaled by sqrt(batch/32)")
    ap.add_argument("--warmup-rows", type=int, default=400)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="take rows in train_serverless.py's order (Random(seed) shuffle of the whole file)")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    os.environ.setdefault("FIREWORKS_CLIENT_SOURCE", "fireworks-training-skill/2.3.0")
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import TrainOnWhat
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")

    def datums(path: Path, n: int):
        out = []
        for line in open(REPO / path):
            r = json.loads(line)
            mi, w = renderer.build_supervised_example(r["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
            t = [int(x) for x in mi.to_ints()]
            w = [float(x) for x in w.tolist()]
            out.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(t[:-1]),
                                 loss_fn_inputs={"target_tokens": t[1:], "weights": w[1:]}))
            if len(out) == n:
                return out
        return out

    if args.shuffle_seed is None:
        train = datums(args.train, args.rows)    # the file is already shuffled: same rows, same order for all
    else:  # identical rows and order to the batch-32 phase of the earlier train_serverless.py run
        all_rows = datums(args.train, 10 ** 9)
        order = random.Random(args.shuffle_seed).sample(range(len(all_rows)), len(all_rows))
        train = [all_rows[i] for i in order[: args.rows]]
    held = datums(args.heldout, args.n_heldout)
    out_dir = REPO / "runs/train/batch_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def say(**kw):
        kw = {"t": time.strftime("%H:%M:%S"), **kw}
        print(json.dumps(kw), flush=True)
        with open(out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")

    def heldout_loss(tc) -> float:
        loss = tokens = 0.0
        for i in range(0, len(held), 16):
            m = tc.forward(held[i:i + 16], "cross_entropy").result().metrics
            loss += m.get("loss:sum", 0.0)
            tokens += m.get("response_tokens", 0.0)
        return loss / max(tokens, 1.0)

    for batch in [int(x) for x in args.configs.split(",")]:
        service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                         base_url=serverless_base_url("https://api.fireworks.ai"))
        enable_serverless_supervised_409_retry(service)
        try:
            for attempt in range(60):
                try:
                    tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=32, alpha=32, seed=0)
                    break
                except Exception as e:  # noqa: BLE001
                    if "429" not in str(e) and "RATE_LIMIT" not in str(e):
                        raise
                    time.sleep(30)
            say(event="session", batch=batch, session=getattr(service, "training_session_id", None))
            before = heldout_loss(tc)
            lr0 = args.lr * math.sqrt(batch / 32)
            pending, done_times, losses = [], [], []
            for s, start in enumerate(range(0, len(train), batch), 1):
                lr = lr0 * min(1.0, (start + batch) / args.warmup_rows)
                fb = tc.forward_backward(train[start:start + batch], "cross_entropy")
                op = tc.optim_step(sdk.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.999, eps=1e-8,
                                                  weight_decay=0.0, grad_clip_norm=1.0))
                pending.append((s, fb, op))
                while len(pending) >= args.inflight or (start + batch >= len(train) and pending):
                    st, f1, f2 = pending.pop(0)
                    m = f1.result().metrics
                    f2.result()
                    done_times.append(time.time())
                    losses.append(m.get("loss:sum", 0.0) / max(m.get("response_tokens", 1.0), 1.0))
                    say(event="step", batch=batch, step=st, loss=round(losses[-1], 4))
            after = heldout_loss(tc)
            settled = done_times[3:]  # skip warm-up steps
            rows_s = (len(settled) - 1) * batch / max(settled[-1] - settled[0], 1e-9) if len(settled) > 1 else 0.0
            res = {"batch": batch, "steps": len(done_times), "rows": len(train), "heldout_before": round(before, 4),
                   "heldout_after": round(after, 4), "rows_per_s_settled": round(rows_s, 2),
                   "hours_for_85k_rows": round(85208 / max(rows_s, 1e-9) / 3600, 1)}
            results.append(res)
            say(event="result", **res)
        finally:
            try:
                service.delete_training_session()
                say(event="session_deleted", batch=batch)
            except Exception as e:  # noqa: BLE001
                say(event="delete_failed", detail=str(e)[:200])
            service.close()
    say(event="comparison", results=results)


if __name__ == "__main__":
    main()
