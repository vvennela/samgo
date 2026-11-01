"""Frozen static repair pilot: shared first response, certificate feedback, matched generic retries.

python -m samgo.eval_certificate_repair --prepare --out runs/audit/certificate-feedback
python -m samgo.eval_certificate_repair --run --out runs/audit/certificate-feedback
"""

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .assembly import FireworksSampler, save
from .certificate_repair import Config, R3, code_identity, evaluate, measure, messages, recheck_trace, repair
from .certificates import digest


def brick(w, length, x, y, z=0, kind="brick"):
    return dict(type=kind, w=w, l=length, x=x, y=y, z=z, color="blue")


def make_cases():
    from .audit_objects import bed_reference
    from .synthesize import synthesize

    def case(name, description, parts, supports, bounds, max_added=8):
        return dict(
            task=dict(id=name, description=description, input=parts, bounds=bounds, max_added=max_added),
            oracle=[dict(op="add", part=p) for p in supports],
        )

    cases = [
        case(
            "shelf_6",
            "Support the existing shelf without moving it.",
            [brick(2, 2, 0, 0), brick(6, 2, 0, 0, 3, "plate")],
            [brick(2, 2, 4, 0)],
            [0, 6, 0, 2, 3],
        ),
        case(
            "shelf_8_shifted",
            "Support the longer, translated shelf without moving it.",
            [brick(2, 2, 7, 4), brick(8, 2, 7, 4, 3, "plate")],
            [brick(2, 2, 13, 4)],
            [7, 15, 4, 6, 3],
        ),
        case(
            "shelf_6_rotated",
            "Support the shelf running along y without moving it.",
            [brick(2, 2, 3, 7), brick(2, 6, 3, 7, 3, "plate")],
            [brick(2, 2, 3, 11)],
            [3, 5, 7, 13, 3],
        ),
        case(
            "table",
            "Add supports beneath the existing three-piece tabletop.",
            [brick(8, 2, 0, 0, 6, "plate"), brick(8, 2, 0, 2, 6, "plate"), brick(2, 4, 3, 0, 7, "plate")],
            [brick(2, 2, x, y, z) for x in (0, 6) for y in (0, 2) for z in (0, 3)],
            [0, 8, 0, 4, 6],
            12,
        ),
        case(
            "bench",
            "Support the existing bench seat and back.",
            [brick(8, 2, 0, 0, 3, "plate"), brick(8, 1, 0, 1, 4)],
            [brick(2, 2, x, 0) for x in (0, 6)],
            [0, 8, 0, 2, 3],
        ),
        case(
            "counterweight_prefix",
            "Make every assembly step stable, including before the counterweight is placed.",
            [brick(2, 2, 2, 0), brick(4, 2, 0, 0, 3, "plate"), brick(2, 2, 2, 0, 4)],
            [brick(2, 2, 0, 0)],
            [0, 4, 0, 2, 3],
        ),
    ]
    bed = bed_reference()
    reference = synthesize(bed, timeout=30)
    if not reference["passed"]:
        raise ValueError("procedural bed reference could not certify; do not run an infeasible pilot")
    supports = reference["build"][len(bed) :]
    cases.append(
        case(
            "bed",
            "Repair support beneath the existing bed; preserve the complete deck and headboard.",
            bed,
            supports,
            [0, 12, 0, 8, 6],
            24,
        )
    )

    def shift(ps):
        return [p | dict(x=p["x"] + 5, y=p["y"] + 3) for p in ps]

    cases.append(
        case(
            "bed_shifted",
            "Repair support beneath the translated bed; preserve all original parts.",
            shift(bed),
            shift(supports),
            [5, 17, 3, 11, 6],
            24,
        )
    )
    for c in cases:
        if evaluate(c["task"], "[]")["passed"]:
            raise ValueError(f"case {c['task']['id']} is already certified")
        if not evaluate(c["task"], json.dumps(c["oracle"]))["passed"]:
            raise ValueError(f"case {c['task']['id']} reference does not pass")
    return cases


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cases.json"
    if path.exists():
        raise ValueError("refusing to overwrite frozen cases")
    cases = make_cases()
    save(
        path,
        dict(
            version=1,
            description="Development probes, not held-out transfer or whole-object generation",
            sha256=digest(cases),
            cases=cases,
        ),
    )
    return cases


def load_cases(root):
    bundle = json.loads((root / "cases.json").read_text())
    if digest(bundle["cases"]) != bundle["sha256"]:
        raise ValueError("frozen case hash mismatch")
    return bundle["cases"]


def experiment(task, sampler, out, config):
    out.mkdir(parents=True, exist_ok=True)
    identity = digest(
        dict(
            task=task,
            sampler=sampler.metadata,
            config=asdict(config),
            code=code_identity(),
            evaluator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    )
    path = out / "first.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("changed experiment identity; use a new output directory")
        if saved["pending"]:
            raise ValueError("interrupted first request: inspect before resampling; it may be billed")
        first = saved["attempt"]
        if first["raw_answer"] is not None:
            checked = evaluate(task, first["raw_answer"], config)
            if checked["passed"] != first["passed"] or checked["candidate"] != first["candidate"]:
                raise ValueError("saved first proposal does not reproduce")
        elif first["passed"]:
            raise ValueError("saved service failure claims success")
    else:
        save(path, dict(identity=identity, pending=True))
        first = measure(task, sampler, messages(task), config)
        save(path, dict(identity=identity, pending=False, attempt=first))
    result = dict(
        case=task["id"],
        first=first["passed"],
        available=first["raw_answer"] is not None,
        first_status=first["status"],
        feedback=False,
        retry=False,
        calls=1,
        tokens=first["tokens"] or 0,
        token_limits=int(first["at_token_limit"]),
        service_errors=int(first["status"] == "service_error"),
        attempts={},
    )
    if first["raw_answer"] is not None:
        # Same first response and caps; stable counterbalancing, no seed support in this sampler.
        arms = ["feedback", "retry"]
        if int(digest(task["id"])[:2], 16) % 2:
            arms.reverse()
        for arm in arms:
            trace = repair(task, sampler, out / arm, config, use_feedback=arm == "feedback", first=first)
            result[arm] = trace["passed"]
            result["attempts"][arm] = len(trace["attempts"])
            extra = trace["attempts"][1:]
            result["calls"] += len(extra)
            result["tokens"] += sum(a["tokens"] or 0 for a in extra)
            result["token_limits"] += sum(a["at_token_limit"] for a in extra)
            result["service_errors"] += sum(a["status"] == "service_error" for a in extra)
    save(out / "result.json", result)
    return result


def summarize(root):
    rows = [json.loads(p.read_text()) for p in sorted((root / "results").glob("*/result.json"))]
    result = dict(
        cases=len(rows),
        available=sum(r["available"] for r in rows),
        first_pass=sum(r["first"] for r in rows),
        feedback_pass=sum(r["feedback"] for r in rows),
        retry_pass=sum(r["retry"] for r in rows),
        feedback_recovered=sum(r["feedback"] and not r["first"] for r in rows),
        retry_recovered=sum(r["retry"] and not r["first"] for r in rows),
        calls=sum(r["calls"] for r in rows),
        output_tokens=sum(r["tokens"] for r in rows),
        token_limits=sum(r["token_limits"] for r in rows),
        service_errors=sum(r["service_errors"] for r in rows),
        results=rows,
    )
    save(root / "summary.json", result)
    return result


def replay(root):
    count = 0
    for case in load_cases(root):
        directory = root / "results" / case["task"]["id"]
        for path in sorted(directory.glob("*/trace.json")):
            trace = json.loads(path.read_text())
            if trace["task"] != case["task"]:
                raise ValueError("saved task differs from frozen case")
            count += recheck_trace(trace)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "summarize", "replay"):
        action.add_argument("--" + name, action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=R3)
    ap.add_argument("--case", help="run one frozen case; subsequent full runs resume it")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()
    if args.prepare:
        print(json.dumps(dict(prepared=len(prepare(args.out)))))
    elif args.replay:
        print(json.dumps(dict(rechecked_attempts=replay(args.out))))
    elif args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
    else:
        cases = load_cases(args.out)
        if args.case:
            cases = [c for c in cases if c["task"]["id"] == args.case]
            if not cases:
                ap.error("case not found")
        sampler = FireworksSampler(args.model)
        config = Config(rounds=args.rounds, max_tokens=args.max_tokens, timeout=args.timeout)
        for case in cases:
            row = experiment(
                copy.deepcopy(case["task"]), sampler, args.out / "results" / case["task"]["id"], config
            )
            print(json.dumps(row), flush=True)
            summarize(args.out)


if __name__ == "__main__":
    main()
