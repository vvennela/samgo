"""Pilot report: baseline vs. round 1 on the frozen test set, plus the go/no-go call.

Go if all three hold (fixed before the run):
  1. baseline pass rate at k = 1 or 2 is between 5% and 90% (the task is learnable),
  2. round 1 improves the overall test pass rate by >= 5 points,
  3. round 1 improves k = 10, which was never trained on (transfer, not memorization).
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    by_k, tasks = defaultdict(list), defaultdict(list)
    codes = defaultdict(int)
    for r in rows:
        by_k[r["k"]].append(r["passed"])
        tasks[r["id"]].append(r["passed"])
        for c in r["codes"]:
            codes[c] += 1
    task_k = {r["id"]: r["k"] for r in rows}
    pass_at_n = defaultdict(list)
    for t, v in tasks.items():
        pass_at_n[task_k[t]].append(any(v))
    n = len(rows)
    return {
        "samples": n,
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "pass_rate_by_k": {k: sum(v) / len(v) for k, v in sorted(by_k.items())},
        "pass_at_n_by_k": {k: sum(v) / len(v) for k, v in sorted(pass_at_n.items())},
        "answered_rate": sum(r["answered"] for r in rows) / n,
        "truncated_rate": sum(r.get("finish") == "length" for r in rows) / n,
        "mean_completion_tokens": sum(len(r.get("completion_ids", [])) for r in rows) / n,
        "failure_codes": dict(sorted(codes.items(), key=lambda x: -x[1])),
    }


def paired_delta(before: list[dict], after: list[dict], k: int | None = None, iters: int = 2000) -> dict:
    """Per-task pass-fraction change with a task-level bootstrap 95% interval."""
    def per_task(rows):
        d = defaultdict(list)
        for r in rows:
            if k is None or r["k"] == k:
                d[r["id"]].append(r["passed"])
        return {t: sum(v) / len(v) for t, v in d.items()}
    b, a = per_task(before), per_task(after)
    ids = sorted(set(b) & set(a))
    if not ids:
        return {}
    diffs = [a[t] - b[t] for t in ids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return {"tasks": len(ids), "delta": sum(diffs) / len(diffs),
            "ci95": [boots[int(0.025 * iters)], boots[int(0.975 * iters)]]}


def main(out: Path) -> None:
    base, after, train = load(out / "base_test/samples.jsonl"), load(out / "r1_test/samples.jsonl"), load(out / "base_train/samples.jsonl")
    metrics = {"baseline_test": summarize(base), "round1_test": summarize(after), "baseline_train": summarize(train)}
    for name in ("train_meta.json",):
        p = out / "round1" / name
        if p.exists():
            metrics["train_meta"] = json.loads(p.read_text())
    log = load(out / "round1/train_log.jsonl")
    if log:
        metrics["train_loss"] = {"first": log[0]["loss"], "last": log[-1]["loss"], "steps": len(log),
                                 "minutes": log[-1]["elapsed_s"] / 60}
    if base and after:
        metrics["delta_overall"] = paired_delta(base, after)
        metrics["delta_by_k"] = {k: paired_delta(base, after, k) for k in sorted({r["k"] for r in base})}

    b = metrics["baseline_test"]
    checks = {}
    if b:
        small = [b["pass_rate_by_k"].get(k, 0) for k in (1, 2)]
        checks["1_learnable"] = any(0.05 <= s <= 0.90 for s in small)
    if "delta_overall" in metrics:
        checks["2_improves_5pts"] = metrics["delta_overall"]["delta"] >= 0.05
        d10 = metrics["delta_by_k"].get(10, {})
        checks["3_transfers_to_k10"] = bool(d10) and d10["delta"] > 0
    metrics["checks"] = checks
    metrics["verdict"] = ("GO" if checks and len(checks) == 3 and all(checks.values())
                          else "NO-GO" if checks else "INCOMPLETE")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    lines = [f"# SAMgo pilot report — verdict: **{metrics['verdict']}**", ""]
    for name, s in (("Baseline (test)", b), ("Round 1 (test)", metrics["round1_test"])):
        if s:
            lines += [f"## {name}", f"- pass rate: {s['pass_rate']:.3f} over {s['samples']} samples",
                      "- by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in s["pass_rate_by_k"].items()),
                      "- pass@n by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in s["pass_at_n_by_k"].items()),
                      f"- answered {s['answered_rate']:.3f}, truncated {s['truncated_rate']:.3f}, "
                      f"mean completion tokens {s['mean_completion_tokens']:.0f}",
                      f"- failure codes: {s['failure_codes']}", ""]
    if "delta_overall" in metrics:
        d = metrics["delta_overall"]
        lines += ["## Change (paired by task, bootstrap 95% CI)",
                  f"- overall: {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"]
        lines += [f"- k={k}: {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                  for k, v in metrics["delta_by_k"].items() if v]
        lines.append("")
    if "train_meta" in metrics:
        lines += ["## Training", f"- {json.dumps(metrics['train_meta'])}", f"- loss: {metrics.get('train_loss')}", ""]
    lines += ["## Checks", *(f"- {k}: {'pass' if v else 'FAIL'}" for k, v in checks.items())]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
