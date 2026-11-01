import copy
import json

import pytest

from samgo.assembly import Sample
from samgo.certificate_repair import Config, messages
from samgo.eval_certificate_repair import experiment, load_cases, prepare, replay, summarize


def task():
    return dict(
        id="shelf",
        description="support shelf",
        max_added=2,
        bounds=[0, 6, 0, 2, 3],
        input=[
            dict(type="brick", w=2, l=2, x=0, y=0, z=0, color="blue"),
            dict(type="plate", w=6, l=2, x=0, y=0, z=3, color="blue"),
        ],
    )


class Sampler:
    metadata = {"source": "test"}

    def __init__(self):
        self.calls = []

    def __call__(self, chat, **kwargs):
        self.calls.append((copy.deepcopy(chat), kwargs))
        return Sample("[]", 2)


def test_matched_budget_shared_first_resume_and_summary(tmp_path):
    s = Sampler()
    out = tmp_path / "results" / "shelf"
    config = Config(rounds=3)
    r = experiment(task(), s, out, config)
    assert len(s.calls) == r["calls"] == 5  # first + two per arm, not two independent first answers
    assert r["tokens"] == 10 and not r["first"] and not r["feedback"] and not r["retry"]
    assert s.calls[1][0][:-1] == s.calls[3][0][:-1]
    assert s.calls[1][1] == s.calls[3][1]
    experiment(task(), s, out, config)
    assert len(s.calls) == 5
    summary = summarize(tmp_path)
    assert summary["cases"] == 1 and summary["calls"] == 5 and summary["output_tokens"] == 10
    with pytest.raises(ValueError, match="changed"):
        experiment(task(), s, out, Config(rounds=2))


def test_frozen_cases_are_feasible_no_references_in_prompts_and_hash_checked(tmp_path):
    cases = prepare(tmp_path)
    assert len(cases) == 8 and len({c["task"]["id"] for c in cases}) == 8
    for c in cases:
        assert json.loads(messages(c["task"])[1]["content"]) == c["task"]
        assert "oracle" not in json.dumps(messages(c["task"]))
    assert load_cases(tmp_path) == cases
    assert replay(tmp_path) == 0
    with pytest.raises(ValueError, match="overwrite"):
        prepare(tmp_path)
    path = tmp_path / "cases.json"
    bundle = json.loads(path.read_text())
    bundle["cases"][0]["task"]["input"][0]["x"] += 1
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cases(tmp_path)


def test_interrupted_first_call_is_not_silently_rebilled(tmp_path):
    s = Sampler()
    experiment(task(), s, tmp_path, Config(rounds=1))
    p = tmp_path / "first.json"
    saved = json.loads(p.read_text())
    saved["pending"] = True
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="interrupted"):
        experiment(task(), s, tmp_path, Config(rounds=1))
    assert len(s.calls) == 1
