"""Held-out validation loss for every saved checkpoint of a run, to decide whether training still helps.

The training loss stopped being a held-out measure: 85k rows hold only 30k distinct structures, so past ~25k
rows over half of each batch repeats a structure already trained on. This scores checkpoints instead on test
rows whose structure never appears in the training file, with the training loss's own definition (mean
negative log-likelihood per answer token, same renderer and weights), through the serverless sampler's prompt
logprobs, so no training session is needed. Paid as prompt tokens; --budget stops before overspending.

Run from the cookbook root with this repo on PYTHONPATH:
    python -m brickgpt.val_loss --run brickgpt-v1 --train data/sft/train_c2.jsonl --test data/sft/eval_500.jsonl
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
PREFILL_PER_M = 1.96  # uncached prompt tokens, Glimmer serverless (fireworks.ai/pricing, fetched Oct 29)


def heldout(train: Path, test: Path, n: int, out: Path) -> list[dict]:
    """The first `n` test rows whose answer (brick structure) is not anywhere in the training file."""
    seen = {json.loads(line)["messages"][-1]["content"] for line in open(train)}
    rows = [r for r in (json.loads(line) for line in open(test)) if r["messages"][-1]["content"] not in seen]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows[:n]))
    return rows[:n]


def checkpoints(run: str) -> list[tuple[str, str]]:
    """(label, sampler path) for every promotable sampler checkpoint of `run`, oldest first."""
    from fireworks.training.sdk import FireworksClient
    import requests

    key, account = os.environ["FIREWORKS_API_KEY"], os.environ.get("FIREWORKS_ACCOUNT_ID", "vvennela")
    sessions = requests.get(f"https://api.fireworks.ai/v1/accounts/{account}/trainingSessions",
                            headers={"Authorization": f"Bearer {key}"}, params={"pageSize": 200}).json()
    fw = FireworksClient(api_key=key, base_url="https://api.fireworks.ai")
    found = []
    for s in sessions.get("trainingSessions", []):
        try:
            listed = fw.list_training_session_checkpoints(s["name"])
        except RuntimeError:  # sessions that never bound a trainer have nothing to list
            continue
        for c in listed:
            name = c.get("checkpointName") or ""
            m = re.fullmatch(r"(run-[0-9a-f]{32})-(.+)", name)
            if c.get("promotable") and m and m.group(2).startswith(run + "-") and "-base-" not in m.group(2):
                rows = re.search(r"-(?:s|partial-)(\d+)-", m.group(2) + "-")
                label = f"{run}@{rows.group(1)}" if rows else m.group(2)
                found.append((c.get("createTime", ""), label, f"{account}/{m.group(1)}/{m.group(2)}"))
    return [(label, path) for _, label, path in sorted(found)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="brickgpt-v1")
    ap.add_argument("--train", type=Path, default=Path("data/sft/train_c2.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("data/sft/eval_500.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budget", type=float, default=4.0, help="USD, counted at the uncached prompt rate")
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()
    load_dotenv(REPO / ".env")

    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import TrainOnWhat
    from training.utils.serverless import serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    from brickgpt.fw_sampler import BASE_MODEL, HF_MODEL

    out_dir = REPO / "runs/val" / args.run
    rows = heldout(REPO / args.train, REPO / args.test, args.n, out_dir / "heldout.jsonl")
    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")
    examples = []
    for r in rows:
        model_input, weights = renderer.build_supervised_example(r["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
        examples.append(([int(t) for t in model_input.to_ints()], [float(x) for x in weights.tolist()]))
    tokens_per_model = sum(len(t) for t, _ in examples)

    results_path = out_dir / "val_loss.jsonl"
    done = {json.loads(line)["label"] for line in open(results_path)} if results_path.exists() else set()
    todo = ([] if args.no_base else [("base", None)]) + checkpoints(args.run)
    todo = [(label, path) for label, path in todo if label not in done]
    print(f"{len(rows)} held-out rows, {tokens_per_model} tokens per model; to score: {[l for l, _ in todo]}")

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url("https://api.fireworks.ai"))
    spent = 0.0
    for label, path in todo:
        cost = tokens_per_model / 1e6 * PREFILL_PER_M
        if spent + cost > args.budget:
            print(f"stop: {label} would bring the spend to ${spent + cost:.2f} (budget ${args.budget})")
            break
        kwargs = {"model_path": path} if path else {"base_model": BASE_MODEL}
        client = service.create_sampling_client(tokenizer=tok, **kwargs)
        futures = [client.compute_logprobs(sdk.ModelInput.from_ints(t)) for t, _ in examples]
        per_row = []
        for (t, w), fut in zip(examples, futures):
            lps = fut.result()
            nll = -sum(lp * wi for lp, wi in zip(lps[1:], w[1:]) if lp is not None)
            per_row.append(nll / sum(w[1:]))
        spent += cost
        mean = sum(per_row) / len(per_row)
        rec = {"label": label, "path": path, "val_loss": round(mean, 5), "per_row": [round(x, 5) for x in per_row]}
        with results_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{label:24s} val loss {mean:.4f}   (spent ~${spent:.2f})", flush=True)

    report(results_path)


def report(results_path: Path) -> None:
    """Each model against the one before it: paired per-row difference and its standard error."""
    recs = [json.loads(line) for line in open(results_path)]
    key = lambda r: -1 if r["label"] == "base" else int(r["label"].rsplit("@", 1)[-1])  # noqa: E731
    recs.sort(key=key)
    print("\nmodel                    val loss   vs previous (paired)")
    prev = None
    for r in recs:
        line = f"{r['label']:24s} {r['val_loss']:.4f}"
        if prev:
            d = [a - b for a, b in zip(r["per_row"], prev["per_row"])]
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
            line += f"   {m:+.4f} +- {se:.4f}  ({'significant' if abs(m) > 2 * se else 'within noise'})"
        print(line)
        prev = r


if __name__ == "__main__":
    main()
