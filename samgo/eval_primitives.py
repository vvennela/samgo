"""Frozen test of the run 6 primitives (hinge a strip, add a leg) on unseen bodies, with the same generic checks and
precise feedback as training, up to three rounds; cases run in parallel.

    python -m samgo.eval_primitives runs/audit/r6-primitives --models r5=<sampler> r6=<sampler>
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .edits import apply_edits
from .free_edit_test import RULES, check, explain
from .primitives import KEEP
from .running_gear_task import FEEDBACK, edits_from


def run_case(c: dict, model: str, rounds: int = 3, max_tokens: int = 8000) -> dict:
    from .assembly import FireworksSampler

    sampler = FireworksSampler(model)
    chat = [{"role": "system", "content": RULES},
            {"role": "user", "content": c["instruction"] + "\nParts:\n" +
             "\n".join(f"{k}: {json.dumps(c['full'][i])}" for k, i in enumerate(c["view"]))}]
    out = {"id": c["id"], "kind": c["kind"], "level": c["level"], "rounds": []}
    for rnd in range(1, rounds + 1):
        t, s = time.time(), None
        entry = {"round": rnd}
        try:
            s = sampler(chat, max_tokens=max_tokens, temperature=0.0 if rnd == 1 else 0.7, timeout=600)
            entry.update(tokens=s.tokens, seconds=round(time.time() - t, 1))
            a = apply_edits(c["full"], edits_from(s.text), local=c["view"])
            v = check(c["full"], a.parts, a.sources, c["requirements"], stand=False)
            keep = {k: x for k, x in v.items() if k in KEEP or k.endswith("_surfaces")}
            entry["passed"] = all(x[0] for x in keep.values())
            entry["failed"] = [k for k, x in keep.items() if not x[0]]
            issues = explain(c["full"], a.parts, a.sources, a.changed, keep, c["view"])
        except Exception as e:  # noqa: BLE001 - unparsable or invalid answers are verdicts too
            entry.update(passed=False, failed=["answer"])
            issues = [f"answer: {type(e).__name__}: {e}"[:200]]
        out["rounds"].append(entry)
        if entry["passed"]:
            break
        if s is not None:
            chat = chat + [{"role": "assistant", "content": s.text},
                           {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))}]
    out["first"] = out["rounds"][0]["passed"]
    out["passed"] = any(r["passed"] for r in out["rounds"])
    return out


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("test", type=Path)
    ap.add_argument("--models", nargs="+", required=True, help="label=sampler path")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cases = json.loads((args.test / "cases.json").read_text())["cases"]
    for spec in args.models:
        label, model = spec.split("=", 1)
        with ThreadPoolExecutor(args.workers) as pool:
            res = list(pool.map(lambda c: run_case(c, model), cases))
        (args.test / f"{label}.json").write_text(json.dumps(res, indent=1))
        summary = {}
        for r in res:
            for key in (r["kind"], f"{r['kind']}/{r['level']}"):
                d = summary.setdefault(key, [0, 0, 0])
                d[0] += r["first"]
                d[1] += r["passed"]
                d[2] += 1
        print(json.dumps({"model": label, "first_try/within_rounds/n": summary}), flush=True)


if __name__ == "__main__":
    main()
