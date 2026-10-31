"""Bounded support repair: propose additions, then independently certify the complete assembly.

python -m samgo.synthesize build.json -o runs/certified-repair
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

from .certificates import DEFAULT_POLICY, MAX_PARTS, Policy, check_certificate, digest, parse_build
from .certify import certify


def support_columns(parts, target, remaining):
    """Permanent 1x1 columns under the failed piece, spreading its support footprint.

    The proposal language deliberately preserves every original part and its position.
    It cannot remove obstructions, introduce temporary jigs, or repair mechanisms.
    """
    p = parts[target]
    height = p["z"]
    count = height // 3 + height % 3
    if not height or count > remaining:
        return
    cells = [(x, y) for x in range(p["x"], p["x"] + p["w"]) for y in range(p["y"], p["y"] + p["l"])]
    underneath = [
        q
        for q in parts
        if q["z"] + (3 if q["type"] == "brick" else 1) == height
        and q["x"] < p["x"] + p["w"]
        and p["x"] < q["x"] + q["w"]
        and q["y"] < p["y"] + p["l"]
        and p["y"] < q["y"] + q["l"]
    ]
    cx = sum(q["x"] + q["w"] / 2 for q in underneath) / len(underneath) if underneath else p["x"] + p["w"] / 2
    cy = sum(q["y"] + q["l"] / 2 for q in underneath) / len(underneath) if underneath else p["y"] + p["l"] / 2
    # This heuristic may use floats: it only orders proposals, never accepts a build.
    cells.sort(key=lambda xy: (-((xy[0] + 0.5 - cx) ** 2 + (xy[1] + 0.5 - cy) ** 2), xy))
    for x, y in cells:
        # The entire vertical column must be free. Contact at its top is allowed.
        if any(
            q["x"] < x + 1
            and x < q["x"] + q["w"]
            and q["y"] < y + 1
            and y < q["y"] + q["l"]
            and q["z"] < height
            for q in parts
        ):
            continue
        column, z = [], 0
        while z < height:
            h = 3 if height - z >= 3 else 1
            column.append(
                dict(type="brick" if h == 3 else "plate", w=1, l=1, x=x, y=y, z=z, color="light_gray")
            )
            z += h
        yield column


def synthesize(raw, *, policy=DEFAULT_POLICY, max_added=24, max_attempts=128, timeout=30.0):
    """Depth-first, bounded feedback search. Only a replayed certificate authorizes output.

    Failure means this restricted search found no certificate, not that the design is impossible.
    This is procedural repair, not learned generation or optimization of the number of parts.
    """
    for name, value, minimum in (("max_added", max_added, 0), ("max_attempts", max_attempts, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    start = time.monotonic()
    result = dict(
        passed=False,
        status="unknown",
        producer="bounded_permanent_support_search_v1",
        original_sha256=digest(raw),
        build=None,
        certificate=None,
        added_indices=[],
        attempts=[],
        specification=policy.spec(),
        physical_validation="not_performed",
        limits=dict(max_added=max_added, max_attempts=max_attempts, timeout=timeout),
    )
    original_count = len(raw) if isinstance(raw, list) else 0
    stack, seen = [copy.deepcopy(raw)], set()
    while stack and len(result["attempts"]) < max_attempts:
        remaining_time = start + timeout - time.monotonic()
        if remaining_time <= 0:
            break
        candidate = stack.pop()
        key = digest(candidate)
        if key in seen:
            continue
        seen.add(key)
        report = certify(candidate, policy=policy, timeout=remaining_time)
        additions = candidate[original_count:] if isinstance(candidate, list) else []
        result["attempts"].append(
            dict(
                build_sha256=key,
                added=len(additions),
                additions=additions,
                status=report["status"],
                failed_part=report["failed_part"],
                issues=report["issues"],
            )
        )
        if report["passed"]:
            checked = check_certificate(candidate, report["certificate"], policy=policy)
            if checked["passed"] and candidate[: len(raw)] == raw:
                result.update(
                    passed=True,
                    status="certified_under_model",
                    build=candidate,
                    certificate=report["certificate"],
                    added_indices=list(range(len(raw), len(candidate))),
                )
            else:
                result.update(status="checker_rejected")
            break
        if report["status"] in ("unsupported", "invalid_geometry"):
            if len(result["attempts"]) == 1:
                result["status"] = report["status"]
            continue
        target = report["failed_part"]
        if target is None or time.monotonic() >= start + timeout:
            continue
        # Parse again at the proposal boundary; unsupported fields cannot enter the search language.
        parse_build(candidate)
        remaining = min(max_added - (len(candidate) - len(raw)), MAX_PARTS - len(candidate))
        proposals = list(support_columns(candidate, target, remaining))
        stack.extend(candidate + column for column in reversed(proposals))
    result["seconds"] = time.monotonic() - start
    if not result["passed"]:
        result["issues"] = [
            "No certificate found within this support language and budget; physical impossibility is not established."
        ]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True, help="new output directory")
    ap.add_argument("--max-added", type=int, default=24)
    ap.add_argument("--max-attempts", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--compression-per-stud", default="250")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output directory already exists; choose a new directory to preserve prior evidence")
    policy = Policy(args.compression_per_stud)
    raw = json.loads(args.build.read_text())
    report = synthesize(
        raw, policy=policy, max_added=args.max_added, max_attempts=args.max_attempts, timeout=args.timeout
    )
    args.out.mkdir(parents=True)
    (args.out / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]:
        from .instructions import write

        (args.out / "build.json").write_text(json.dumps(report["build"], indent=2) + "\n")
        write(
            report["build"],
            args.out / "instructions",
            certificate=report["certificate"],
            certificate_policy=policy,
        )
    print(
        json.dumps(
            dict(
                status=report["status"],
                attempts=len(report["attempts"]),
                added=len(report["added_indices"]),
                seconds=report["seconds"],
            ),
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
