"""Produce exact rational witnesses with Z3, then check them independently.

python -m samgo.certify prove build.json -o certificate-report.json
python -m samgo.certify check build.json certificate-report.json
"""

import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path

from .certificates import (
    DEFAULT_POLICY,
    VERSION,
    CertificateError,
    Policy,
    Unsupported,
    check_certificate,
    check_path,
    contacts,
    digest,
    parse_build,
    rational,
    validate_geometry,
    validate_order,
)


def solve_forces(pieces, placed, policy, deadline):
    import z3

    def q(value):
        f = Fraction(value)
        return z3.RealVal(f"{f.numerator}/{f.denominator}")

    cs = contacts(pieces, placed)
    solver = z3.SolverFor("QF_LRA")
    fs = [z3.Real(f"force_{k}") for k in range(len(cs))]
    totals = {i: [[], [], []] for i in placed}
    studs = {}
    for contact, f in zip(cs, fs):
        up, down, x, y, stud = contact
        solver.add(f >= 0)
        if stud is not None:
            studs.setdefault(stud, []).append(f)
        for i, sign in ((up, 1), (down, -1)):
            if i is None:
                continue
            p = pieces[i]
            totals[i][0].append(sign * f)
            totals[i][1].append(sign * f * q(x - p.x - Fraction(p.w, 2)))
            totals[i][2].append(sign * f * q(y - p.y - Fraction(p.l, 2)))
    for values in studs.values():
        solver.add(z3.Sum(values) <= q(rational(policy.compression_per_stud)))
    for i, (force, tx, ty) in totals.items():
        p = pieces[i]
        solver.add(z3.Sum(force) == p.w * p.l * p.h, z3.Sum(tx) == 0, z3.Sum(ty) == 0)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "resource deadline exhausted"
    solver.set(timeout=max(1, int(remaining * 1000)))
    result = solver.check()
    if result == z3.unsat:
        return None, "this assembly prefix has no compression-only equilibrium under the declared capacity"
    if result != z3.sat:
        return None, "solver returned unknown: " + solver.reason_unknown()
    model = solver.model()
    values = [model.eval(f, model_completion=True) for f in fs]
    return [str(Fraction(v.numerator_as_long(), v.denominator_as_long())) for v in values], None


def certify(raw, *, policy=DEFAULT_POLICY, order=None, timeout=20.0):
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    start = time.monotonic()
    report = dict(
        passed=False,
        status="unknown",
        specification=policy.spec(),
        physical_validation="not_performed",
        certificate=None,
        completed_steps=0,
        issues=[],
    )

    def finish():
        report["seconds"] = time.monotonic() - start
        return report

    try:
        pieces = parse_build(raw)
        validate_geometry(pieces)
    except Unsupported as e:
        report.update(status="unsupported", issues=[str(e)])
        return finish()
    except (ValueError, TypeError, KeyError) as e:
        report.update(status="invalid_geometry", issues=[str(e)])
        return finish()
    order = (
        order
        if order is not None
        else sorted(range(len(pieces)), key=lambda i: (pieces[i].z, pieces[i].y, pieces[i].x, i))
    )
    try:
        validate_order(order, len(pieces))
        certificate = dict(
            version=VERSION,
            build_sha256=digest(raw),
            spec_sha256=digest(policy.spec()),
            order=order,
            stages=[],
        )
        placed = []
        for index in order:
            check_path(pieces, placed, index)
            placed.append(index)
            witness, issue = solve_forces(pieces, placed, policy, start + timeout)
            if issue:
                report.update(
                    issues=[
                        f"step {len(placed)} (part {index}): {issue}",
                        "No certificate found for this order; this does not prove physical impossibility.",
                    ]
                )
                return finish()
            certificate["stages"].append(dict(part=index, forces=witness))
            report["completed_steps"] += 1
        checked = check_certificate(raw, certificate, policy=policy)
        if not checked["passed"]:
            report.update(status="checker_rejected", issues=checked["issues"])
            return finish()
        report.update(passed=True, status="certified_under_model", certificate=certificate)
    except CertificateError as e:
        report.update(status="not_certified", issues=[str(e)])
    return finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prove = sub.add_parser("prove")
    prove.add_argument("build", type=Path)
    prove.add_argument("-o", "--out", type=Path, required=True)
    prove.add_argument("--timeout", type=float, default=20)
    prove.add_argument("--order", type=Path, help="JSON list of original part indices")
    prove.add_argument("--instructions", type=Path, help="export the exact certified order only on success")
    check = sub.add_parser("check")
    check.add_argument("build", type=Path)
    check.add_argument("report", type=Path)
    for command in (prove, check):
        command.add_argument(
            "--compression-per-stud",
            default="250",
            help="declared abstract force capacity; not a measured strength",
        )
    args = ap.parse_args()
    parts = json.loads(args.build.read_text())
    policy = Policy(args.compression_per_stud)
    if args.command == "prove":
        order = json.loads(args.order.read_text()) if args.order else None
        result = certify(parts, order=order, policy=policy, timeout=args.timeout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if result["passed"] and args.instructions:
            from .instructions import write

            write(parts, args.instructions, certificate=result["certificate"], certificate_policy=policy)
        print(json.dumps({k: v for k, v in result.items() if k != "certificate"}, indent=2))
    else:
        report = json.loads(args.report.read_text())
        result = check_certificate(parts, report.get("certificate"), policy=policy)
        print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
