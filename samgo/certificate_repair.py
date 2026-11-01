"""Learned, add-only static repairs with bounded certificate feedback and auditable retries.

python -m samgo.certificate_repair task.json -o runs/learned-static-repair
"""

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .assembly import FireworksSampler, save
from .certificates import DEFAULT_POLICY, MAX_PARTS, check_certificate, digest, parse_build
from .certify import certify
from .edits import apply_edits
from .parts import COLORS, FOOTPRINTS
from .running_gear_task import edits_from

VERSION = "learned-static-certificate-repair-v1"
R3 = "vvennela/run-6a81b005f4ba4a039e8dda1760edf3b9/running-gear-r3-tuned-ef4d7fa6"
CONTROL = (
    "Try again and independently check your proposal against every stated requirement. "
    "Return the complete JSON add-edit list against the original input; it replaces your previous answer."
)


@dataclass(frozen=True)
class Config:
    rounds: int = 3
    max_tokens: int = 4000
    timeout: float = 60
    certificate_timeout: float = 10
    retry_temperature: float = 0.7

    def __post_init__(self):
        for name in ("rounds", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout", "certificate_timeout", "retry_temperature"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
            if v < 0 or (name != "retry_temperature" and v == 0):
                raise ValueError(f"invalid {name}")


def code_identity():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "certificate_repair.py",
            "certify.py",
            "certificates.py",
            "edits.py",
            "parts.py",
            "assembly.py",
            "running_gear_task.py",
        )
    }


def validate_task(task):
    if set(task) != {"id", "description", "input", "max_added", "bounds"}:
        raise ValueError("task requires exactly id, description, input, max_added, bounds")
    if not isinstance(task["id"], str) or not task["id"] or not isinstance(task["description"], str):
        raise ValueError("task id and description must be strings")
    parse_build(task["input"])
    if type(task["max_added"]) is not int or not 0 <= task["max_added"] <= MAX_PARTS - len(task["input"]):
        raise ValueError("invalid maximum number of additions")
    b = task["bounds"]
    if (
        not isinstance(b, list)
        or len(b) != 5
        or any(type(v) is not int for v in b)
        or b[0] >= b[1]
        or b[2] >= b[3]
        or b[4] <= 0
    ):
        raise ValueError("bounds must be integer [xmin, xmax, ymin, ymax, maximum_top_z]")


def messages(task, policy=DEFAULT_POLICY):
    validate_task(task)
    system = (
        "Repair a static LEGO assembly by adding permanent supports. Preserve every original part and position. "
        'Return ONLY a JSON list of edits: [{"op":"add","part":{"type":"brick",'
        '"w":1,"l":1,"x":0,"y":0,"z":0,"color":"light_gray"}}]. '
        "That example illustrates syntax, not a solution. Only add operations are allowed. Each attempt replaces "
        "all previous proposed additions and starts from the ORIGINAL input. Parts must have exactly those fields. "
        "Types: brick (height 3) or plate (height 1). Coordinates and dimensions are integers in studs/plates; "
        "z >= 0. Allowed footprints, either orientation: "
        + json.dumps(FOOTPRINTS)
        + ". Colors: "
        + ", ".join(COLORS)
        + ". Bounds are [xmin, xmax, ymin, ymax, maximum_top_z]. Every added part must satisfy "
        "xmin <= x, x+w <= xmax, ymin <= y, y+l <= ymax, and z+height <= maximum_top_z. "
        "The bounds constrain the entire solid, not just its origin. "
        "The final assembly must be stud-connected, touch ground, and have no solid-envelope overlaps. "
        "The assembly order is fixed: ascending (z,y,x,input index). At EVERY prefix, every piece must balance "
        "its weight and torques using compressive contacts only. Weight is w*l*height at its center. Stud "
        "contact points are at 1/4 and 3/4 within shared stud cells; ground contacts are inset 1/4 from corners. "
        f"Total compression per shared stud is at most {policy.compression_per_stud} abstract force units. "
        "Studs cannot pull or hold cantilevers in tension. Each new piece must be insertable vertically from above "
        "without crossing existing solids. No temporary supports or subassemblies. Do not return a proof: the "
        "checker will generate and independently replay a certificate. Fewer added parts are preferable, but "
        "validity is required. These constraints describe an idealized static model, not measured strength."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(task, separators=(",", ":"))},
    ]


def evaluate(task, answer, config=Config(), policy=DEFAULT_POLICY):
    """One authoritative candidate, task contract, and independently replayed certificate."""
    validate_task(task)
    result = dict(
        passed=False,
        status="invalid_answer",
        issues=[],
        edits=None,
        candidate=None,
        certificate=None,
        failed_part=None,
        completed_steps=0,
    )
    try:
        edits = edits_from(answer)
        result["edits"] = edits
        if len(edits) > task["max_added"]:
            raise ValueError(f"at most {task['max_added']} added parts are allowed")
        candidate = apply_edits(task["input"], edits, allowed_ops=frozenset({"add"})).parts
        result["candidate"] = candidate
        pieces = parse_build(candidate)
        xmin, xmax, ymin, ymax, top = task["bounds"]
        for i, p in enumerate(pieces[len(task["input"]) :], len(task["input"])):
            if not (
                xmin <= p.x and p.x + p.w <= xmax and ymin <= p.y and p.y + p.l <= ymax and p.z + p.h <= top
            ):
                raise ValueError(f"added part {i} is outside the allowed support bounds")
        report = certify(candidate, policy=policy, timeout=config.certificate_timeout)
        result.update({k: report[k] for k in ("status", "issues", "failed_part", "completed_steps")})
        if report["passed"]:
            checked = check_certificate(candidate, report["certificate"], policy=policy)
            if checked["passed"]:
                result.update(passed=True, certificate=report["certificate"])
            else:
                result.update(status="checker_rejected", issues=checked["issues"])
    except (ValueError, TypeError, KeyError) as e:
        result.update(status="invalid_answer", issues=[str(e)])
    return result


def feedback(attempt):
    detail = {k: attempt[k] for k in ("status", "issues", "failed_part", "completed_steps")}
    if attempt.get("failed_part") is not None and attempt.get("candidate") is not None:
        detail["failed_piece"] = attempt["candidate"][attempt["failed_part"]]
    return (
        "Certificate/task feedback:\n"
        + json.dumps(detail)
        + "\n"
        + CONTROL
        + " A failed prefix is not proof that no solution exists. Fix the reported issue without "
        "introducing overlaps or leaving other prefixes unstable."
    )


def measure(task, sampler, chat, config=Config(), policy=DEFAULT_POLICY, temperature=0.0):
    started = time.monotonic()
    attempt = dict(
        messages=copy.deepcopy(chat),
        temperature=temperature,
        raw_answer=None,
        tokens=None,
        at_token_limit=False,
        passed=False,
        status="service_error",
        issues=[],
        failed_part=None,
        completed_steps=0,
        candidate=None,
        certificate=None,
        edits=None,
    )
    try:
        sample = sampler(
            copy.deepcopy(chat), max_tokens=config.max_tokens, timeout=config.timeout, temperature=temperature
        )
    except Exception as e:
        attempt.update(error_kind=type(e).__name__, issues=["sampler unavailable or timed out"])
    else:
        attempt.update(
            raw_answer=sample.text,
            tokens=sample.tokens,
            at_token_limit=sample.tokens is not None and sample.tokens >= config.max_tokens,
        )
        attempt.update(evaluate(task, sample.text, config, policy))
    attempt["seconds"] = time.monotonic() - started
    return attempt


def recheck_trace(trace, policy=DEFAULT_POLICY):
    """Re-evaluate saved proposals; never treat a persisted success flag as evidence."""
    config = Config(**trace["config"])
    accepted = None
    for attempt in trace["attempts"]:
        if attempt["raw_answer"] is None:
            if attempt["passed"]:
                raise ValueError("saved service failure claims success")
            continue
        checked = evaluate(trace["task"], attempt["raw_answer"], config, policy)
        if checked["passed"] != attempt["passed"] or checked["candidate"] != attempt["candidate"]:
            raise ValueError("saved proposal does not reproduce its verdict/candidate")
        if checked["passed"]:
            if not check_certificate(checked["candidate"], attempt["certificate"], policy=policy)["passed"]:
                raise ValueError("saved certificate failed independent replay")
            accepted = checked["candidate"]
    if trace["complete"] and (trace["passed"] != (accepted is not None) or trace["accepted"] != accepted):
        raise ValueError("saved final verdict/build does not match checked proposals")
    return len(trace["attempts"])


def repair(task, sampler, out, config=Config(), *, use_feedback=True, first=None, policy=DEFAULT_POLICY):
    """Bounded retries. Failed proposals never become an accepted build; resume never resamples completion."""
    chat = messages(task, policy)
    identity = digest(
        dict(
            task=task,
            sampler=sampler.metadata,
            config=asdict(config),
            feedback=use_feedback,
            policy=policy.spec(),
            code=code_identity(),
            first=first,
        )
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trace.json"
    if path.exists():
        trace = json.loads(path.read_text())
        if trace["identity"] != identity:
            raise ValueError("refusing resume with changed task, code, sampler, policy or protocol")
        if trace.get("pending"):
            raise ValueError(
                "interrupted sampling request: inspect trace before starting a new run; it may be billed"
            )
        if trace["task"] != task or trace["config"] != asdict(config):
            raise ValueError("saved task/protocol differs from requested run")
        recheck_trace(trace, policy)
        if trace["complete"]:
            return trace
    else:
        trace = dict(
            version=VERSION,
            identity=identity,
            task=task,
            sampler=sampler.metadata,
            config=asdict(config),
            use_feedback=use_feedback,
            code_sha256=code_identity(),
            specification=policy.spec(),
            attempts=[],
            passed=False,
            complete=False,
            pending=False,
            accepted=None,
            physical_validation="not_performed",
            shared_first=first is not None,
        )
        if first is not None:
            if first["messages"] != chat or first["temperature"] != 0.0:
                raise ValueError("shared first response belongs to a different prompt or temperature")
            first = copy.deepcopy(first)
            if first["raw_answer"] is not None:
                first.update(evaluate(task, first["raw_answer"], config, policy))
            else:
                first.update(passed=False, certificate=None, candidate=None, status="service_error")
            trace["attempts"].append(first)
        save(path, trace)
    for attempt in trace["attempts"]:
        if attempt["raw_answer"] is not None:
            chat += [
                {"role": "assistant", "content": attempt["raw_answer"]},
                {"role": "user", "content": feedback(attempt) if use_feedback else CONTROL},
            ]
    while not any(a["passed"] for a in trace["attempts"]) and len(trace["attempts"]) < config.rounds:
        trace["pending"] = True
        save(path, trace)
        attempt = measure(
            task,
            sampler,
            chat,
            config,
            policy,
            temperature=config.retry_temperature if trace["attempts"] else 0.0,
        )
        trace["attempts"].append(attempt)
        trace["pending"] = False
        save(path, trace)
        if attempt["raw_answer"] is not None:
            chat += [
                {"role": "assistant", "content": attempt["raw_answer"]},
                {"role": "user", "content": feedback(attempt) if use_feedback else CONTROL},
            ]
    accepted = next((a for a in trace["attempts"] if a["passed"]), None)
    if accepted:
        # Replay from raw edits, not a persisted passed flag or saved candidate.
        checked = evaluate(task, accepted["raw_answer"], config, policy)
        if not checked["passed"]:
            raise ValueError("accepted proposal failed replay; no output exported")
        from .instructions import write

        save(out / "build.json", checked["candidate"])
        save(out / "certificate.json", dict(certificate=checked["certificate"]))
        write(
            checked["candidate"],
            out / "instructions",
            certificate=checked["certificate"],
            certificate_policy=policy,
        )
        trace.update(passed=True, accepted=checked["candidate"])
    trace["complete"] = True
    save(path, trace)
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--model", default=R3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--no-feedback", action="store_true")
    args = ap.parse_args()
    trace = repair(
        json.loads(args.task.read_text()),
        FireworksSampler(args.model),
        args.out,
        Config(rounds=args.rounds, max_tokens=args.max_tokens, timeout=args.timeout),
        use_feedback=not args.no_feedback,
    )
    print(json.dumps(dict(passed=trace["passed"], attempts=len(trace["attempts"]))))
    raise SystemExit(0 if trace["passed"] else 1)


if __name__ == "__main__":
    main()
