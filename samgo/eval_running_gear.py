"""First-try pass rate on held-out running-gear problems, for base Glimmer or a fine-tuned checkpoint.

Each problem is a validation row's first turn (system + user); the model's answer is parsed as edits, applied to
the same car, and checked (structure, function incl. wheels held, stability). Same rows for every model, so the
numbers compare. Run from the cookbook root with this repo on PYTHONPATH.

    python -m samgo.eval_running_gear --n 10                                  # base Glimmer
    python -m samgo.eval_running_gear --n 10 --model vvennela/run-.../name    # a fine-tuned checkpoint
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .running_gear_task import FEEDBACK, apply, axles_of, check, edits_from, explain, frame_problem, pose, pose_open

REPO = Path(__file__).resolve().parent.parent
MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"


def problems(n: int, data: str = "data/running_gear", split: str = "val") -> list[dict]:
    """The first-turn problems of a split, spread over as many different vehicles as possible."""
    rows = [json.loads(line) for line in open(REPO / data / f"{split}.jsonl")]
    rows = [r for r in rows if r["turns"] == 1]
    seen, spread = set(), []
    for r in rows:  # one problem per vehicle first, then the rest
        uid = r["base"].rsplit("_", 1)[0]
        if uid not in seen:
            seen.add(uid)
            spread.append(r)
    rows = (spread + [r for r in rows if r not in spread])[:n]
    out = []
    bases = "runs/bases_shell" if "frame" in data else "runs/bases"
    for r in rows:
        parts = json.loads((REPO / f"{bases}/{r['base']}.json").read_text())
        target = np.load(REPO / f"{bases}/{r['base']}.npz")["target"]
        if "frame" in data:
            problem, partial = frame_problem(parts, target, r["task"])
        else:
            edges = axles_of(parts)
            strip = {"front": [edges[0]], "rear": [edges[-1]], "both": edges}[r["task"]]
            problem, partial = (pose_open if "open" in data else pose)(parts, target, strip), False
        if problem is not None:
            out.append({"row": r, "problem": problem, "target": target, "partial": partial})
    return out


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="sampler checkpoint path; default base Glimmer")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data", default="data/running_gear", help="data/running_gear_open for the open task")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=1, help="answers per problem: 1 = first try; more = with feedback")
    ap.add_argument("--split", default="val", help="val to measure; train to collect the model's own mistakes")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import get_text_content
    from training.utils.serverless import serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url("https://api.fireworks.ai"))
    client = service.create_sampling_client(tokenizer=tok, **({"model_path": args.model} if args.model
                                                               else {"base_model": MODEL}))
    params = sdk.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                stop=renderer.get_stop_sequences())
    probs = problems(args.n, args.data, args.split)
    chats = [list(p["row"]["messages"][:2]) for p in probs]
    results = [{"base": p["row"]["base"], "task": p["row"]["task"], "passed": False, "valid_format": False,
                "round": None, "issues": []} for p in probs]
    tokens_out = 0
    for rnd in range(1, args.rounds + 1):
        todo = [k for k, r in enumerate(results) if not r["passed"]]
        if not todo:
            break
        futures = {k: client.sample(prompt=renderer.build_generation_prompt(chats[k]), num_samples=1,
                                    sampling_params=params) for k in todo}
        for k, fut in futures.items():
            p, rec = probs[k], results[k]
            try:
                toks = list(fut.result(timeout=900).sequences[0].tokens)
                tokens_out += len(toks)
                msg, _ = renderer.parse_response(toks)
                text = get_text_content(msg)
                chats[k].append({"role": "assistant", "content": text})
                edits = edits_from(text)
                ok, issues = explain(p["problem"], edits, partial=p["partial"])  # the messages the model trains on
                rec |= {"valid_format": True, "passed": ok, "issues": issues[:3], "edits": len(edits), "answer": edits,
                        "reference": p["problem"]["answer"], "round": rnd if ok else None}
            except Exception as e:  # noqa: BLE001
                issues = [f"could not use the answer: {type(e).__name__}: {str(e)[:120]}"]
                rec |= {"passed": False, "issues": issues}
                if chats[k][-1]["role"] != "assistant":
                    chats[k].append({"role": "assistant", "content": ""})
            if not rec["passed"]:
                chats[k].append({"role": "user", "content": FEEDBACK.format(
                    issues="\n".join(f"- {i}" for i in rec["issues"]))})
        print(json.dumps({"round": rnd, "passed_so_far": sum(r["passed"] for r in results)}), flush=True)
    passed = sum(r["passed"] for r in results)
    fmt = sum(r["valid_format"] for r in results)
    first = sum(r["round"] == 1 for r in results)
    summary = {"model": args.model or "base", "n": len(results), "passed_first_try": first,
               "passed_within_rounds": passed, "rounds": args.rounds, "valid_edit_lists": fmt,
               "output_tokens": tokens_out}
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
