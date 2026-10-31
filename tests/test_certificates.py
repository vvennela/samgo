import copy
import sys
import time

import pytest

from samgo.certificates import (
    CertificateError,
    Policy,
    check_certificate,
    check_forces,
    check_path,
    parse_build,
)
from samgo.certify import certify, solve_forces


def brick(x=0, y=0, z=0, w=2, length=2, kind="brick"):
    return dict(type=kind, w=w, l=length, x=x, y=y, z=z, color="blue")


def bridge():
    # Deliberately unsorted input: certificate indices must refer to original parts.
    return [brick(z=3, w=8, kind="plate"), brick(), brick(x=6)]


def test_exact_certificate_is_checked_without_solver(monkeypatch):
    parts = bridge()
    report = certify(parts)
    assert report["passed"] and report["physical_validation"] == "not_performed"
    assert report["certificate"]["order"] == [1, 2, 0]
    monkeypatch.setitem(sys.modules, "z3", None)
    assert check_certificate(parts, report["certificate"])["passed"]


def test_bed_fixture_certifies_every_step():
    import json
    from pathlib import Path

    parts = json.loads((Path(__file__).parent / "fixtures/certifiable_bed.json").read_text())
    report = certify(parts)
    assert report["passed"] and report["completed_steps"] == 12
    assert len(report["certificate"]["stages"]) == len(parts)


@pytest.mark.parametrize("mutation", ["build", "force", "float", "stage", "duplicate", "spec", "negative"])
def test_forged_or_mismatched_witnesses_are_rejected(mutation):
    parts = bridge()
    cert = copy.deepcopy(certify(parts)["certificate"])
    if mutation == "build":
        parts[0]["x"] += 1
    elif mutation == "force":
        cert["stages"][0]["forces"][0] = "999"
    elif mutation == "float":
        cert["stages"][0]["forces"][0] = 0.0
    elif mutation == "stage":
        cert["stages"].pop(0)
    elif mutation == "duplicate":
        cert["order"] = [1, 1, 0]
    elif mutation == "spec":
        cert["spec_sha256"] = "fake"
    else:
        cert["stages"][0]["forces"][0] = "-1"
    assert not check_certificate(parts, cert)["passed"]


def test_certificate_cannot_relax_caller_capacity():
    parts = bridge()
    cert = certify(parts)["certificate"]
    assert not check_certificate(parts, cert, policy=Policy("1"))["passed"]
    low = certify(parts, policy=Policy("1"))
    assert not low["passed"] and low["status"] == "unknown"
    assert "capacity" in low["issues"][0]


def test_exact_checker_rejects_sub_float_tolerance_error_and_excess_capacity():
    from fractions import Fraction
    from samgo.certificates import digest

    parts = bridge()
    cert = certify(parts)["certificate"]
    wrong = copy.deepcopy(cert)
    value = Fraction(wrong["stages"][0]["forces"][0]) + Fraction(1, 10**50)
    wrong["stages"][0]["forces"][0] = str(value)
    assert not check_certificate(parts, wrong)["passed"]
    # Even forging the expected policy hash cannot make an overloaded force witness valid.
    cert["spec_sha256"] = digest(Policy("1").spec())
    checked = check_certificate(parts, cert, policy=Policy("1"))
    assert not checked["passed"] and "capacity exceeded" in checked["issues"][0]


@pytest.mark.parametrize("extra", [dict(type="wheel"), dict(type="hinge_top"), dict(type="axle")])
def test_unsupported_mechanisms_are_never_dropped(extra):
    r = certify(bridge() + [extra])
    assert not r["passed"] and r["status"] == "unsupported" and r["certificate"] is None


def test_unmodeled_fields_fail_closed():
    r = certify([brick() | {"external_load": 1000}])
    assert r["status"] == "unsupported" and not r["passed"]


@pytest.mark.parametrize(
    "parts", [[], [brick(), brick()], [brick(), brick(x=2)], [brick(z=3)], [brick(w=True)]]
)
def test_invalid_empty_overlap_disconnected_floating_and_boolean(parts):
    assert not certify(parts)["passed"]


def test_continuous_insertion_rejects_obstacle_above_destination():
    pieces = parse_build([brick(), brick(z=3)])
    # Final positions are disjoint, but the lower brick cannot descend through the upper one.
    with pytest.raises(CertificateError, match="blocked"):
        check_path(pieces, [1], 0)
    check_path(pieces, [0], 1)


def test_final_equilibrium_does_not_hide_an_unstable_construction_prefix():
    parts = [brick(x=2), brick(z=3, w=4, kind="plate"), brick(x=2, z=4)]
    pieces = parse_build(parts)
    witness, issue = solve_forces(pieces, [0, 1, 2], Policy(), time.monotonic() + 5)
    assert issue is None
    check_forces(pieces, [0, 1, 2], witness, Policy())
    result = certify(parts)
    assert not result["passed"] and result["completed_steps"] == 1
    assert "does not prove physical impossibility" in result["issues"][1]


def test_translation_and_quarter_turn_preserve_certifiability():
    parts = bridge()
    moved = [p | dict(x=p["y"] - 20, y=p["x"] + 100, w=p["l"], l=p["w"]) for p in parts]
    assert certify(moved)["passed"]


def test_timeout_is_unknown_not_unsat_or_success():
    result = certify(bridge(), timeout=1e-9)
    assert result["status"] == "unknown" and not result["passed"] and result["certificate"] is None


def test_instruction_export_uses_exact_proved_order_and_rejects_wrong_build(tmp_path):
    from samgo.instructions import write

    parts = bridge()
    certificate = certify(parts)["certificate"]
    doc = write(parts, tmp_path / "good", certificate=certificate)
    assert "3 parts in 3 steps" in doc.read_text()
    assert "gravity equilibrium" in doc.read_text()
    assert (doc.parent / "assembly_certificate.json").exists()
    with pytest.raises(ValueError, match="certificate rejected"):
        write(parts[::-1], tmp_path / "bad", certificate=certificate)
    assert not (tmp_path / "bad").exists()
