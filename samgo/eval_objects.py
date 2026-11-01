"""Controlled first-answer vs verifier-feedback vs equal-budget retry experiments.

Run `--prepare` once to freeze cases. Subsequent evaluations never regenerate their geometry.
"""

import argparse
import copy
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .assembly import FireworksSampler, digest, save
from .audit_objects import inspect_case, make_cases, messages
from .running_gear_task import FEEDBACK, edits_from

MODELS = {
    "base": None,
    "r3": "vvennela/run-6a81b005f4ba4a039e8dda1760edf3b9/running-gear-r3-tuned-ef4d7fa6",
}
CONTROL = (
    "Try again and independently check whether your proposed assembly satisfies all the stated requirements. "
    "Return the complete JSON edit list against the original input; it replaces your previous answer."
)


def measure(case, sampler, chat, *, max_tokens=4000, timeout=60, temperature=0.0):
    start = time.monotonic()
    result = dict(
        messages=copy.deepcopy(chat),
        temperature=temperature,
        raw_answer=None,
        edits=None,
        passed=False,
        tokens=None,
        codes=[],
        issues=[],
        candidate=None,
    )
    try:
        sample = sampler(chat, max_tokens=max_tokens, timeout=timeout, temperature=temperature)
        result.update(
            raw_answer=sample.text,
            tokens=sample.tokens,
            at_token_limit=bool(sample.tokens and sample.tokens >= max_tokens),
        )
    except Exception as e:
        result.update(
            error_kind=type(e).__name__, codes=["service_error"], issues=["sampler unavailable or timed out"]
        )
        result["seconds"] = time.monotonic() - start
        return result
    try:
        edits = edits_from(sample.text)
        result["edits"] = edits
        result.update(inspect_case(case, edits))
    except ValueError as e:
        result.update(codes=["parse_error"], issues=[str(e)])
    result["seconds"] = time.monotonic() - start
    return result


def experiment(case, sampler, out: Path, protocol):
    path = out / f"{case['id']}.json"
    identity = digest(dict(case=case, protocol=protocol, sampler=sampler.metadata))
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["identity"] != identity:
            raise ValueError("refusing to resume with changed case/protocol/model")
        if prior.get("complete"):
            return prior
    else:
        prior = dict(
            identity=identity,
            case=case["id"],
            family=case["family"],
            mode=case["mode"],
            sampler=sampler.metadata,
            protocol=protocol,
            complete=False,
        )
    chat = messages(case)
    if "first" not in prior:
        prior["first"] = measure(
            case, sampler, chat, max_tokens=protocol["max_tokens"], timeout=protocol["timeout"]
        )
        save(path, prior)
    first = prior["first"]
    if not first["passed"] and first["raw_answer"] is not None:
        # Both arms start from the very same failed response. One extra response each, same temperature/cap.
        prefix = chat + [{"role": "assistant", "content": first["raw_answer"]}]
        feedback = FEEDBACK.format(issues="\n".join(f"- {s}" for s in first["issues"]))
        # Counterbalance call order by case id; avoid systematically giving one arm a warmer service.
        arms = [("feedback", feedback), ("retry", CONTROL)]
        if int(digest(case["id"])[:2], 16) % 2:
            arms.reverse()
        for arm, text in arms:
            if arm not in prior:
                prior[arm] = measure(
                    case,
                    sampler,
                    prefix + [{"role": "user", "content": text}],
                    max_tokens=protocol["max_tokens"],
                    timeout=protocol["timeout"],
                    temperature=0.7,
                )
                prior[arm]["repeated_edits"] = (
                    prior[arm]["edits"] is not None and prior[arm]["edits"] == first["edits"]
                )
                prior[arm]["resolved_codes"] = sorted(set(first["codes"]) - set(prior[arm]["codes"]))
                save(path, prior)
    prior["complete"] = True
    save(path, prior)
    return prior


def summarize(root):
    rows = []
    for p in root.glob("results/*/*.json"):
        r = json.loads(p.read_text())
        if r.get("complete"):
            rows.append(r | {"model": p.parent.name})
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        data = [r for r in rows if r["model"] == model]
        available = [r for r in data if r["first"]["raw_answer"] is not None]
        failed = [r for r in available if not r["first"]["passed"]]
        summary[model] = dict(
            cases=len(data),
            available=len(available),
            first_pass=sum(r["first"]["passed"] for r in available),
            failed_first=len(failed),
            empty_first_answers=sum(r["first"]["raw_answer"] == "" for r in data),
            first_service_errors=sum("service_error" in r["first"]["codes"] for r in data),
            token_limit_responses=sum(
                bool(r.get(a, {}).get("at_token_limit")) for r in data for a in ("first", "feedback", "retry")
            ),
            feedback_recovered=sum(r.get("feedback", {}).get("passed", False) for r in failed),
            retry_recovered=sum(r.get("retry", {}).get("passed", False) for r in failed),
            repeated_after_feedback=sum(r.get("feedback", {}).get("repeated_edits", False) for r in failed),
            output_tokens=sum(
                a.get("tokens") or 0
                for r in data
                for k, a in r.items()
                if k in ("first", "feedback", "retry")
            ),
            first_error_counts=dict(Counter(c for r in failed for c in r["first"]["codes"])),
        )
    save(root / "summary.json", {"models": summary, "complete_cases": len(rows)})
    return summary


def run_model(name, cases, root, protocol):
    out = root / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    sampler = FireworksSampler(MODELS[name])
    for case in cases:
        record = experiment(case, sampler, out, protocol)
        print(
            json.dumps(
                dict(
                    model=name,
                    case=case["id"],
                    first=record["first"]["passed"],
                    feedback=record.get("feedback", {}).get("passed"),
                    retry=record.get("retry", {}).get("passed"),
                    codes=record["first"]["codes"],
                )
            ),
            flush=True,
        )


def render_gallery(root):
    """Export reference and observed proposals with explicit provenance, without inference."""
    from .viewer3d import page
    from .instructions import write

    bundle = json.loads((root / "cases.json").read_text())
    if digest(bundle["cases"]) != bundle["sha256"]:
        raise ValueError("frozen case hash mismatch")
    models = []
    out = root / "visuals"
    out.mkdir(exist_ok=True)
    for case in bundle["cases"]:
        models.append(
            (
                case["id"] + " — procedural reference",
                case["reference"],
                [("source", "reference construction, not learned output"), ("task", case["mode"])],
            )
        )
        for path in sorted((root / "results").glob(f"*/{case['id']}.json")):
            record = json.loads(path.read_text())
            arm = "first" if record["first"]["passed"] else "feedback"
            answer = record.get(arm, record["first"])
            if answer.get("candidate") is not None:
                models.append(
                    (
                        case["id"] + f" — {path.parent.name} {arm}",
                        answer["candidate"],
                        [
                            ("verdict", "PASS" if answer["passed"] else "FAIL"),
                            ("issues", ", ".join(answer["codes"])),
                        ],
                    )
                )
        if case["id"] in ("train_6_axles", "bed_four_legs"):
            write(case["reference"], out / (case["id"] + "_reference"))
    (out / "gallery.html").write_text(page(models, simulate=False))
    return out / "gallery.html"


def verify_saved(root):
    """Recheck every parsed proposal offline, including failed ones; never resample."""
    bundle = json.loads((root / "cases.json").read_text())
    if digest(bundle["cases"]) != bundle["sha256"]:
        raise ValueError("frozen case hash mismatch")
    cases = {c["id"]: c for c in bundle["cases"]}
    checks = []
    for path in sorted((root / "results").glob("*/*.json")):
        record = json.loads(path.read_text())
        for arm in ("first", "feedback", "retry"):
            saved = record.get(arm, {})
            if saved.get("edits") is None:
                continue
            actual = inspect_case(cases[record["case"]], saved["edits"])
            checks.append(
                dict(
                    record=str(path),
                    arm=arm,
                    agrees=all(actual[k] == saved[k] for k in ("passed", "codes", "candidate")),
                )
            )
    save(root / "replay-check.json", checks)
    if any(not c["agrees"] for c in checks):
        raise ValueError("saved proposals disagree with current checker; see replay-check.json")
    return {"replayed_proposals": len(checks), "all_agree": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/audit/object-diagnostics"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--models", nargs="+", default=["r3"], help=f"names ({', '.join(MODELS)}) or name=sampler_path")
    ap.add_argument("--include-local-fixtures", action="store_true")
    ap.add_argument("--single-axle", action="store_true")
    ap.add_argument("--render", action="store_true", help="render saved proposals and references offline")
    ap.add_argument("--summarize", action="store_true", help="summarize saved complete results offline")
    ap.add_argument("--verify-saved", action="store_true", help="recheck parsed proposals offline")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()
    if args.verify_saved:
        print(json.dumps(verify_saved(args.out)))
        return
    if args.render:
        print(render_gallery(args.out))
        return
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
        return
    if args.prepare:
        if (args.out / "cases.json").exists():
            raise ValueError("refusing to overwrite frozen cases; use a new output directory")
        make_cases(args.out, include_local_fixtures=args.include_local_fixtures, single_axle=args.single_axle)
        return
    for spec in args.models:  # name=path registers a newly trained sampler
        if "=" in spec:
            name, path = spec.split("=", 1)
            MODELS[name] = path
    args.models = [m.split("=", 1)[0] for m in args.models]
    unknown = set(args.models) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    bundle = json.loads((args.out / "cases.json").read_text())
    cases = bundle["cases"]
    if digest(cases) != bundle["sha256"]:
        raise ValueError("frozen case hash mismatch")
    if args.cases:
        unknown = set(args.cases) - {c["id"] for c in cases}
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.cases]
    protocol = dict(
        version=1,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        first_temperature=0.0,
        correction_temperature=0.7,
        extra_responses_per_arm=1,
        scope="behavioral diagnostic; does not expose or establish hidden reasoning",
        cases_hash=bundle["sha256"],
    )
    save(args.out / "protocol.json", protocol)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_model, name, cases, args.out, protocol) for name in args.models]
        for job in as_completed(jobs):
            job.result()
    print(json.dumps(summarize(args.out), indent=2))


if __name__ == "__main__":
    main()
