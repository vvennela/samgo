"""Warm-trainer speed comparison of batch sizes: one session, warm-started from the earlier run's snapshot,
alternating blocks (32, 64, 32, 64) so shared-load drift hits both. The first steps after each switch are
not timed. The session is always deleted.

    python -u brickgpt/speed_compare.py
"""

import argparse
import json
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
    ap.add_argument("--state", type=Path, default=Path("runs/train/brickgpt-v1/state.json"))
    ap.add_argument("--blocks", default="32:13,64:12,32:12,64:13")
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--skip-after-switch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--inflight", type=int, default=2)
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

    state = json.loads((REPO / args.state).read_text())
    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")
    rows = [json.loads(line) for line in open(REPO / args.train)]
    order = random.Random(0).sample(range(len(rows)), len(rows))  # train_serverless.py's order
    cursor = state.get("rows", 0)  # continue where the snapshot left off

    def datum(i):
        mi, w = renderer.build_supervised_example(rows[i]["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
        t = [int(x) for x in mi.to_ints()]
        w = [float(x) for x in w.tolist()]
        return sdk.Datum(model_input=sdk.ModelInput.from_ints(t[:-1]), loss_fn_inputs={"target_tokens": t[1:], "weights": w[1:]})

    blocks = [(int(b), int(n)) for b, n in (x.split(":") for x in args.blocks.split(","))]
    plan = [(blocks[0][0], "warmup")] * args.warmup_steps
    for k, (b, n) in enumerate(blocks):
        plan += [(b, f"block{k}")] * n
    need = sum(b for b, _ in plan)
    data = [datum(i) for i in order[cursor:cursor + need]]
    out_dir = REPO / "runs/train/speed_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    def say(**kw):
        kw = {"t": time.strftime("%H:%M:%S"), **kw}
        print(json.dumps(kw), flush=True)
        with open(out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")

    say(event="prepared", start_row=cursor, rows=need, blocks=blocks)
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=serverless_base_url("https://api.fireworks.ai"))
    enable_serverless_supervised_409_retry(service)
    try:
        tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=32, alpha=32)
        if state.get("adapter_path"):
            tc.load_adapter(state["adapter_path"]).result()
        say(event="session", session=getattr(service, "training_session_id", None), warm_start=bool(state.get("adapter_path")))
        pending, done, pos, prev_tag, since_switch = [], [], 0, None, 0
        for b, tag in plan:
            since_switch = since_switch + 1 if tag == prev_tag else 1
            prev_tag = tag
            fb = tc.forward_backward(data[pos:pos + b], "cross_entropy")
            op = tc.optim_step(sdk.AdamParams(learning_rate=args.lr * (b / 32) ** 0.5, beta1=0.9, beta2=0.999, eps=1e-8,
                                              weight_decay=0.0, grad_clip_norm=1.0))
            pos += b
            pending.append((b, tag, since_switch, fb, op))
            while len(pending) >= args.inflight or (pos >= need and pending):
                pb, ptag, ps, f1, f2 = pending.pop(0)
                m = f1.result().metrics
                f2.result()
                done.append({"batch": pb, "tag": ptag, "since_switch": ps, "t": time.time(),
                             "loss": m.get("loss:sum", 0.0) / max(m.get("response_tokens", 1.0), 1.0)})
                say(event="step", batch=pb, tag=ptag, n=len(done), loss=round(done[-1]["loss"], 4))
        # Rows/s per timed step: time since the previous completed step, excluding warm-up and post-switch steps.
        per = {32: [], 64: []}
        per_block = {}
        for prev, cur in zip(done, done[1:]):
            if cur["tag"] == "warmup" or cur["since_switch"] <= args.skip_after_switch:
                continue
            dt = cur["t"] - prev["t"]
            per[cur["batch"]].append((cur["batch"], dt))
            per_block.setdefault(cur["tag"], []).append((cur["batch"], dt))
        rate = lambda xs: sum(b for b, _ in xs) / max(sum(d for _, d in xs), 1e-9)
        res = {f"batch{b}": {"rows_per_s": round(rate(xs), 2), "timed_steps": len(xs),
                             "hours_for_85k": round(85208 / max(rate(xs), 1e-9) / 3600, 1)} for b, xs in per.items() if xs}
        res["per_block_rows_per_s"] = {k: round(rate(v), 2) for k, v in per_block.items()}
        say(event="result", **res)
    finally:
        try:
            service.delete_training_session()
            say(event="session_deleted")
        except Exception as e:  # noqa: BLE001
            say(event="delete_failed", detail=str(e)[:200])
        service.close()


if __name__ == "__main__":
    main()
