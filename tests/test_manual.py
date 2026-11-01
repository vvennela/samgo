import copy

from samgo.manual import MAX_STEP, certify_manual, check
from samgo.skateboard import build_skateboard


def test_skateboard_manual_is_certified_and_rechecks():
    parts = build_skateboard(length=16, width=4)["parts"]
    r = certify_manual(parts, timeout=300)
    assert r["passed"], r["issues"]
    assert all(len(s["parts"]) <= MAX_STEP for s in r["steps"])
    assert check(parts, r)["passed"]
    listed = sorted(i for s in r["steps"] for i in s["parts"])
    assert listed == list(range(len(parts)))               # every part exactly once


def test_tampered_certificates_are_rejected():
    parts = build_skateboard(length=16, width=4)["parts"]
    r = certify_manual(parts, timeout=300)
    forged = copy.deepcopy(r)
    step = next(s for s in forged["steps"] if s["kind"] == "layer")
    step["forces"][0] = "1000000"                           # a force that breaks the balance
    assert not check(parts, forged)["passed"]
    other = copy.deepcopy(parts)
    other[0]["x"] += 1                                      # the certificate belongs to a different model
    assert not check(other, r)["passed"]
