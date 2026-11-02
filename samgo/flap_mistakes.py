"""Run 8: Glimmer's own wrong answers on training wings, turned into rows that teach acting on feedback.

Run 7 repeats its answer, byte for byte, after precise feedback once it is confident (the 747 flap: same answer all
three rounds). Synthetic mistakes only teach fixing mistakes we imagined; these rows use the mistakes it actually
makes. For each training case (samgo.flap_data, never a test object): sample the model greedily, as at test time;
if the answer fails the generic checks, the row is the prompt, its own answer, the precise feedback, then the
verified reference answer (which always differs from the wrong one).

    PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m samgo.flap_mistakes data/flaps_r8 --model <r7 sampler> --n 600
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .flap_data import prompt
from .running_gear_task import FEEDBACK


def one(c: dict, model: str) -> dict:
    from .eval_primitives import run_case

    return {"case": c, "result": run_case(c, model, rounds=1)}


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="a samgo.flap_data output folder (cases.jsonl)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cases = [json.loads(line) for line in (args.data / "cases.jsonl").open()]
    random.Random(81).shuffle(cases)
    cases = cases[:args.n]
    with ThreadPoolExecutor(args.workers) as pool:
        got = list(pool.map(lambda c: one(c, args.model), cases))
    rows, stats = [], {"n": len(got), "passed": 0, "wrong": 0, "by_failure": {}}
    for g in got:
        c, r = g["case"], g["result"]
        first = r["rounds"][0]
        if first["passed"]:
            stats["passed"] += 1
            continue
        if not first.get("text") or not first.get("feedback"):
            continue
        stats["wrong"] += 1
        for f in first["failed"]:
            stats["by_failure"][f] = stats["by_failure"].get(f, 0) + 1
        chat = prompt(c) + [
            {"role": "assistant", "content": first["text"]},
            {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in first["feedback"]))},
            {"role": "assistant", "content": json.dumps(c["answer"])}]
        rows.append({"messages": chat, "family": "flap_r8_own", "mode": c["level"], "case": c["id"], "turns": 2})
    (args.data / "own_mistakes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.data / "own_mistakes_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
