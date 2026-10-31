import copy

import pytest

from samgo.certificates import Policy, check_certificate
from samgo.certify import certify
from samgo.synthesize import synthesize


def cantilever():
    return [
        dict(type="brick", w=2, l=2, x=0, y=0, z=0, color="blue"),
        dict(type="plate", w=6, l=2, x=0, y=0, z=3, color="blue"),
    ]


def test_repair_preserves_input_and_certifies_every_prefix():
    raw = cantilever()
    before = copy.deepcopy(raw)
    assert not certify(raw)["passed"]
    result = synthesize(raw, max_added=2)
    assert result["passed"]
    assert raw == before == result["build"][: len(raw)]
    assert 0 < len(result["added_indices"]) <= 2
    assert check_certificate(result["build"], result["certificate"])["passed"]
    assert len(result["certificate"]["stages"]) == len(result["build"])
    assert not check_certificate(raw, result["certificate"])["passed"]


@pytest.mark.parametrize("limits", [{"max_added": 0}, {"max_attempts": 1}, {"timeout": 1e-12}])
def test_exhaustion_has_no_accepted_build_or_certificate(limits):
    result = synthesize(cantilever(), **limits)
    assert not result["passed"] and result["status"] == "unknown"
    assert result["build"] is None and result["certificate"] is None


def test_unsupported_mechanisms_are_not_stripped():
    raw = cantilever() + [dict(type="wheel", axis="y", x=0, y=0, z=0)]
    result = synthesize(raw)
    assert result["status"] == "unsupported" and not result["passed"]
    assert len(result["attempts"]) == 1


def test_existing_valid_build_is_returned_without_additions():
    result = synthesize(cantilever()[:1])
    assert result["passed"] and not result["added_indices"]


@pytest.mark.parametrize("raw", [None, [], {"type": "brick"}])
def test_malformed_build_has_no_accepted_output(raw):
    result = synthesize(raw)
    assert result["status"] == "invalid_geometry" and result["build"] is None


def test_original_audit_bed_can_be_repaired_without_replacing_its_parts():
    from samgo.audit_objects import bed_reference

    raw = bed_reference()
    result = synthesize(raw, timeout=30)
    assert result["passed"] and result["build"][: len(raw)] == raw
    assert len(result["added_indices"]) <= 24
    assert check_certificate(result["build"], result["certificate"])["passed"]


def test_repair_obeys_caller_capacity():
    result = synthesize(cantilever(), policy=Policy("1"), max_added=2, max_attempts=8)
    assert not result["passed"]


def test_false_producer_success_is_independently_rejected(monkeypatch):
    import samgo.synthesize as module

    fake = certify(cantilever()[:1])
    monkeypatch.setattr(module, "certify", lambda *args, **kwargs: fake)
    result = synthesize(cantilever())
    assert result["status"] == "checker_rejected"
    assert result["build"] is None and result["certificate"] is None


@pytest.mark.parametrize("options", [{"max_added": True}, {"max_attempts": 0}, {"timeout": float("inf")}])
def test_invalid_limits(options):
    with pytest.raises(ValueError):
        synthesize(cantilever(), **options)
