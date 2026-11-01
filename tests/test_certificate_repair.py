import copy
import json

import pytest

from samgo.assembly import Sample
from samgo.certificate_repair import Config, evaluate, measure, messages, repair
from samgo.certificates import check_certificate


def task():
    return dict(
        id="cantilever",
        description="Support the existing shelf",
        max_added=2,
        bounds=[0, 6, 0, 2, 3],
        input=[
            dict(type="brick", w=2, l=2, x=0, y=0, z=0, color="blue"),
            dict(type="plate", w=6, l=2, x=0, y=0, z=3, color="blue"),
        ],
    )


def solution():
    return json.dumps([dict(op="add", part=dict(type="brick", w=2, l=2, x=4, y=0, z=0, color="light_gray"))])


class Sampler:
    metadata = {"source": "test", "model": "scripted"}

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def __call__(self, chat, **kwargs):
        self.calls.append((copy.deepcopy(chat), kwargs))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return Sample(answer, 20)


def test_feedback_repairs_from_original_and_exports_bound_certificate(tmp_path):
    raw = task()
    before = copy.deepcopy(raw)
    s = Sampler(["[]", solution()])
    r = repair(raw, s, tmp_path)
    assert r["passed"] and len(r["attempts"]) == 2
    assert raw == before and r["accepted"][:2] == raw["input"]
    assert "failed_piece" in s.calls[1][0][-1]["content"]
    assert '"failed_part": 1' in s.calls[1][0][-1]["content"]
    cert = json.loads((tmp_path / "certificate.json").read_text())["certificate"]
    assert check_certificate(r["accepted"], cert)["passed"]
    assert (tmp_path / "instructions/instructions.md").exists()
    repair(raw, s, tmp_path)
    assert len(s.calls) == 2
    with pytest.raises(ValueError, match="changed"):
        repair(raw, s, tmp_path, Config(rounds=2))


def test_failures_never_publish_an_accepted_build(tmp_path):
    s = Sampler(["[]"] * 3)
    r = repair(task(), s, tmp_path)
    assert len(s.calls) == 3 and not r["passed"] and r["accepted"] is None
    assert not (tmp_path / "build.json").exists()
    assert not (tmp_path / "instructions").exists()


def test_control_and_feedback_share_first_but_only_feedback_gets_verdict(tmp_path):
    t = task()
    s = Sampler(["[]", "[]", "[]"])
    first = measure(t, s, messages(t))
    a = repair(t, s, tmp_path / "feedback", Config(rounds=2), first=first)
    b = repair(t, s, tmp_path / "retry", Config(rounds=2), first=first, use_feedback=False)
    assert a["attempts"][0] == b["attempts"][0]
    assert s.calls[1][0][:-1] == s.calls[2][0][:-1]
    assert s.calls[1][1] == s.calls[2][1]
    assert "Certificate/task feedback" in s.calls[1][0][-1]["content"]
    assert "failed_part" not in s.calls[2][0][-1]["content"]


def test_invalid_edits_and_bounds_cannot_game_certificate():
    for text in (
        "not json",
        '[{"op":"remove","index":1}]',
        solution().replace('"x": 4', '"x": 6'),
        solution().replace('"type": "brick"', '"type": "wheel"'),
    ):
        r = evaluate(task(), text)
        assert not r["passed"] and r["certificate"] is None
    t = task() | {"max_added": 0}
    assert not evaluate(t, solution())["passed"]


def test_sampler_error_is_bounded_and_secrets_are_not_logged(tmp_path):
    s = Sampler([RuntimeError("secret-credential"), "[]"])
    r = repair(task(), s, tmp_path, Config(rounds=2))
    assert not r["passed"] and len(s.calls) == 2
    assert r["attempts"][0]["status"] == "service_error"
    assert "secret-credential" not in (tmp_path / "trace.json").read_text()


def test_forged_shared_success_is_rechecked(tmp_path):
    s = Sampler(["[]"])
    first = measure(task(), s, messages(task()))
    first.update(passed=True, candidate=task()["input"], certificate={})
    r = repair(task(), s, tmp_path, Config(rounds=1), first=first)
    assert not r["passed"] and not (tmp_path / "build.json").exists()


def test_independent_checker_rejects_producer_success(monkeypatch):
    from samgo import certificate_repair as module

    monkeypatch.setattr(
        module,
        "certify",
        lambda *args, **kwargs: dict(
            passed=True,
            status="certified_under_model",
            certificate={},
            issues=[],
            failed_part=None,
            completed_steps=3,
        ),
    )

    assert evaluate(task(), solution())["status"] == "checker_rejected"


def test_prompt_defines_the_positional_bounds_and_full_solid_constraints():
    system = messages(task())[0]["content"]
    assert "[xmin, xmax, ymin, ymax, maximum_top_z]" in system
    assert "x+w <= xmax" in system and "y+l <= ymax" in system
    assert "z+height <= maximum_top_z" in system


def test_resume_rejects_tampered_success_and_does_not_resample(tmp_path):
    s = Sampler(["[]"])
    repair(task(), s, tmp_path, Config(rounds=1))
    p = tmp_path / "trace.json"
    trace = json.loads(p.read_text())
    trace["passed"] = True
    p.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="final verdict"):
        repair(task(), s, tmp_path, Config(rounds=1))
    assert len(s.calls) == 1


@pytest.mark.parametrize(
    "kwargs", [{"rounds": True}, {"rounds": 0}, {"timeout": float("nan")}, {"retry_temperature": -1}]
)
def test_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
