import copy
import json

import numpy as np
import pytest

from samgo.assembly import Config, ReplaySampler, Sample, assemble
from samgo.parts import COLOR_INDEX, HEIGHTS
from samgo.running_gear_task import check
from samgo.technic import structural
from tests.test_technic import car


@pytest.fixture
def reference():
    parts = car()[2:]
    parts[:0] = [dict(type='technic_brick', w=2, l=1, x=x, y=y, z=0, color='black', hole='round')
                 for x in (0, 6) for y in (1, 2)]
    parts += [dict(type='plate', w=8, l=2, x=0, y=1, z=z, color='red') for z in (7, 8)]
    for p in parts:
        if p['type'] == 'wheel':
            p.pop('width')
            p['size'] = '30.4x14'
    target = np.zeros((8, 5, 9), dtype=int)
    for p in structural(parts):
        target[p['x']:p['x'] + p['w'], p['y']:p['y'] + p['l'], p['z']:p['z'] + HEIGHTS[p['type']]] = COLOR_INDEX[p['color']]
    return parts, target


def answers(parts):
    return [json.dumps([{'op': 'add', 'part': p} for p in parts
                        if p['type'] in ('technic_brick', 'axle', 'wheel') and p['x'] in (x, x + 1)])
            for x in (0, 6)]


class ScriptedSampler:
    metadata = {'source': 'test_fixture', 'model': 'scripted'}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def __call__(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return Sample(response, 10)


def test_feedback_repair_and_replay_preserve_exact_accepted_assembly(reference, tmp_path):
    parts, target = reference
    sampler = ScriptedSampler(['not JSON', *answers(parts)])
    final, trace = assemble(parts, target, tmp_path / 'live', sampler)
    assert trace['passed'] and check(final, target)[0]
    assert trace['steps'][0]['attempts'][0]['candidate'] is None
    assert trace['steps'][0]['attempts'][1]['messages'][-1]['role'] == 'user'
    assert len(final) == len(parts)
    assert trace['output_tokens'] == 30
    assert json.loads((tmp_path / 'live/accepted.json').read_text()) == final
    replayed, replay_trace = assemble(parts, target, tmp_path / 'replay', ReplaySampler(tmp_path / 'live/trace.json'))
    assert replay_trace['passed'] and replay_trace['sampler']['source'] == 'replay'
    assert replayed == final


def test_failed_second_step_keeps_first_step_and_never_uses_reference_fallback(reference, tmp_path):
    parts, target = reference
    first, _ = answers(parts)
    # A duplicate of the already placed wheel is a well-formed, physically invalid proposal.
    bad = json.dumps([{'op': 'add', 'part': next(p for p in parts if p['type'] == 'wheel')}])
    final, trace = assemble(parts, target, tmp_path, ScriptedSampler([first, bad]), Config(rounds=1))
    assert not trace['passed'] and len(trace['steps']) == 2
    assert final == trace['steps'][0]['accepted']
    assert trace['steps'][1]['attempts'][0]['candidate'] != final
    assert len([p for p in final if p['type'] == 'wheel']) == 2
    assert not check(final, target)[0]


def test_service_error_has_bounded_retries_and_safe_diagnostics(reference, tmp_path):
    parts, target = reference
    sampler = ScriptedSampler([TimeoutError('secret token must not appear'), RuntimeError('private')])
    final, trace = assemble(parts, target, tmp_path, sampler, Config(rounds=2))
    assert len(sampler.messages) == 2 and not trace['passed']
    assert final == json.loads((tmp_path / 'input.json').read_text())
    assert 'secret token' not in (tmp_path / 'trace.json').read_text()
    assert [a['error_kind'] for a in trace['steps'][0]['attempts']] == ['TimeoutError', 'RuntimeError']


def test_replay_refuses_a_different_input(reference, tmp_path):
    parts, target = reference
    assemble(parts, target, tmp_path / 'source', ScriptedSampler(answers(parts)))
    # Replay adapter itself compares the entire message history, not just a task label.
    replay = ReplaySampler(tmp_path / 'source/trace.json')
    with pytest.raises(ValueError, match='prompt mismatch'):
        replay([{'role': 'user', 'content': 'different car'}])


def test_joint_mode_is_checked_as_a_whole_car(reference, tmp_path):
    parts, target = reference
    edits = [e for answer in answers(parts) for e in json.loads(answer)]
    final, trace = assemble(parts, target, tmp_path, ScriptedSampler([json.dumps(edits)]), Config(sequential=False))
    assert trace['passed'] and check(final, target)[0]
    assert len(trace['steps']) == 1 and trace['steps'][0]['partial'] is False


def test_infeasible_partial_reference_uses_a_verified_joint_task(reference, tmp_path):
    parts, target = reference
    parts = [p for p in parts if p['z'] < 7]
    target = target[:, :, :7]
    edits = [e for answer in answers(parts) for e in json.loads(answer)]
    final, trace = assemble(parts, target, tmp_path, ScriptedSampler([json.dumps(edits)]))
    assert trace['passed'] and check(final, target)[0]
    assert trace['task_mode'] == 'joint' and trace['config']['sequential'] is True
    assert 'partial reference' in trace['task_mode_reason']
    assert len(trace['steps']) == 1


def test_replay_rejects_target_changes_even_when_prompts_would_match(reference, tmp_path):
    parts, target = reference
    assemble(parts, target, tmp_path / 'recorded', ScriptedSampler(answers(parts)))
    changed = target.copy()
    changed[0, 0, 0] = 1
    final, trace = assemble(parts, changed, tmp_path / 'replayed', ReplaySampler(tmp_path / 'recorded/trace.json'))
    assert not trace['passed'] and not final and not trace['steps']
    assert 'assembly/target mismatch' in trace['error']
