"""Pilot report: baseline vs. round 1 on the frozen test set, plus the go/no-go call.

Go if all three hold (fixed before the run):
  1. some level's baseline pass rate is in the 10-90% band (the task is learnable),
  2. round 1 improves the pass rate on the trained levels by >= 5 points,
  3. round 1 improves the held-out transfer level, never trained on, with the 95% CI above zero.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

LEVELS = ["1", "2", "5", "10", "20", "all"]


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
        by_k[r["level"]].append(r["passed"])
        tasks[r["id"]].append(r["passed"])
        for c in r["codes"]:
            codes[c] += 1
    task_k = {r["id"]: r["level"] for r in rows}
    pass_at_n = defaultdict(list)
    for t, v in tasks.items():
        pass_at_n[task_k[t]].append(any(v))
    n = len(rows)
    return {
        "samples": n,
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "pass_rate_by_level": {k: sum(v) / len(v) for k, v in sorted(by_k.items(), key=lambda x: LEVELS.index(x[0]))},
        "pass_at_n_by_level": {k: sum(v) / len(v) for k, v in sorted(pass_at_n.items(), key=lambda x: LEVELS.index(x[0]))},
        "answered_rate": sum(r["answered"] for r in rows) / n,
        "truncated_rate": sum(r.get("finish") == "length" for r in rows) / n,
        "mean_completion_tokens": sum(len(r.get("completion_ids", [])) for r in rows) / n,
        "failure_codes": dict(sorted(codes.items(), key=lambda x: -x[1])),
    }


def paired_delta(before: list[dict], after: list[dict], levels: set | None = None, iters: int = 2000) -> dict:
    """Per-task pass-fraction change with a task-level bootstrap 95% interval."""
    def per_task(rows):
        d = defaultdict(list)
        for r in rows:
            if levels is None or r["level"] in levels:
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
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else {}
    metrics["plan"] = plan
    if base and after:
        metrics["delta_overall"] = paired_delta(base, after)
        metrics["delta_by_level"] = {lv: paired_delta(base, after, {lv}) for lv in LEVELS if any(r["level"] == lv for r in base)}
        if plan.get("train_levels"):
            metrics["delta_trained_levels"] = paired_delta(base, after, set(plan["train_levels"]))

    b = metrics["baseline_test"]
    checks = {}
    if b:
        checks["1_learnable"] = any(0.10 <= p <= 0.90 for p in b["pass_rate_by_level"].values())
    if "delta_trained_levels" in metrics:
        checks["2_improves_5pts_on_trained_levels"] = metrics["delta_trained_levels"]["delta"] >= 0.05
        t = plan.get("transfer_level")
        if t:
            dt = metrics["delta_by_level"].get(t, {})
            # Transfer counts only if the whole 95% interval is above zero.
            checks[f"3_transfers_to_level_{t}"] = bool(dt) and dt["ci95"][0] > 0
    metrics["checks"] = checks
    metrics["verdict"] = ("GO" if len(checks) == 3 and all(checks.values())
                          else "NO-GO" if checks else "INCOMPLETE")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    lines = [f"# SAMgo pilot report — verdict: **{metrics['verdict']}**", "", f"Plan: {json.dumps(plan)}", ""]
    for name, s in (("Baseline (test)", b), ("Round 1 (test)", metrics["round1_test"])):
        if s:
            lines += [f"## {name}", f"- pass rate: {s['pass_rate']:.3f} over {s['samples']} samples",
                      "- by level: " + ", ".join(f"{k}: {v:.3f}" for k, v in s["pass_rate_by_level"].items()),
                      "- pass@n by level: " + ", ".join(f"{k}: {v:.3f}" for k, v in s["pass_at_n_by_level"].items()),
                      f"- answered {s['answered_rate']:.3f}, truncated {s['truncated_rate']:.3f}, "
                      f"mean completion tokens {s['mean_completion_tokens']:.0f}",
                      f"- failure codes: {s['failure_codes']}", ""]
    if "delta_overall" in metrics:
        d = metrics["delta_overall"]
        lines += ["## Change (paired by task, bootstrap 95% CI)",
                  f"- overall: {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"]
        if "delta_trained_levels" in metrics:
            d = metrics["delta_trained_levels"]
            lines.append(f"- trained levels {plan['train_levels']}: {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        lines += [f"- level {k}: {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                  for k, v in metrics["delta_by_level"].items() if v]
        lines.append("")
    if "train_meta" in metrics:
        lines += ["## Training", f"- {json.dumps(metrics['train_meta'])}", f"- loss: {metrics.get('train_loss')}", ""]
    lines += ["## Checks", *(f"- {k}: {'pass' if v else 'FAIL'}" for k, v in checks.items())]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
