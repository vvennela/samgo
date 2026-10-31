"""The checked candidate must be the same assembly the edit executor produces."""

import copy

import numpy as np
import pytest

from samgo.parts import COLOR_INDEX, HEIGHTS
from samgo.running_gear_task import TASK_FRAME, apply, check, edits_from, explain
from samgo.technic import structural
from tests.test_technic import car


@pytest.fixture
def problem():
    parts = car()
    target = np.zeros((8, 5, 7), dtype=int)
    for p in structural(parts):
        target[p['x']:p['x'] + p['w'], p['y']:p['y'] + p['l'],
               p['z']:p['z'] + HEIGHTS[p['type']]] = COLOR_INDEX[p['color']]
    return {'full': parts, 'local': [0, 1, 2, 3], 'target': target}


@pytest.mark.parametrize('index', [-1, 4, True, 1.2, '1'])
def test_invalid_indices_are_rejected_by_both_paths(problem, index):
    edits = [{'op': 'remove', 'index': index}]
    with pytest.raises(ValueError):
        apply(problem, edits)
    ok, messages = explain(problem, edits)
    assert not ok and messages


@pytest.mark.parametrize('edits', [None, {}, [None], [{'op': 'move'}],
                                 [{'op': 'add', 'part': None}],
                                 [{'op': 'remove', 'index': 0, 'typo': 1}]])
def test_invalid_edit_shapes_fail_cleanly(problem, edits):
    with pytest.raises(ValueError):
        apply(problem, edits)
    assert not explain(problem, edits)[0]


@pytest.mark.parametrize('ops', [('remove', 'set'), ('set', 'remove'), ('set', 'set')])
def test_repeated_operations_on_one_part_are_ambiguous(problem, ops):
    edits = [{'op': op, 'index': 0, **({'part': problem['full'][0]} if op == 'set' else {})} for op in ops]
    with pytest.raises(ValueError):
        apply(problem, edits)
    assert not explain(problem, edits)[0]


def test_frame_task_enforces_add_only(problem):
    problem['task'] = TASK_FRAME
    edits = [{'op': 'set', 'index': 0, 'part': problem['full'][0]}]
    with pytest.raises(ValueError):
        apply(problem, edits)
    assert not explain(problem, edits)[0]


@pytest.mark.parametrize('kind', ['unchanged', 'missing_body', 'overlap', 'blocked_axle'])
def test_feedback_agrees_with_checking_applied_candidate(problem, kind):
    edits = {
        'unchanged': [],
        'missing_body': [{'op': 'remove', 'index': 3}],
        'overlap': [{'op': 'add', 'part': problem['full'][3]}],
        'blocked_axle': [{'op': 'set', 'index': 0, 'part': problem['full'][0] | {'type': 'brick'}}],
    }[kind]
    expected = check(apply(problem, edits), problem['target'])[0]
    ok, messages = explain(problem, edits)
    assert ok == expected
    assert ok or messages


def test_edits_use_original_local_numbering_and_do_not_alias_inputs(problem):
    original = copy.deepcopy(problem['full'])
    edits = [{'op': 'remove', 'index': 0},
             {'op': 'set', 'index': 1, 'part': original[1] | {'color': 'red'}},
             {'op': 'add', 'part': original[0]}]
    result = apply(problem, edits)
    assert result[0] == original[1] | {'color': 'red'}
    assert result[-1] == original[0]
    result[0]['color'] = 'blue'
    result[-1]['x'] = 100
    assert problem['full'] == original
    assert edits[1]['part']['color'] == 'red'


def test_empty_edit_list_parses():
    assert edits_from('[]') == []
    assert edits_from('```json\n[]\n```') == []


def test_malformed_parts_fail_without_crashing_feedback(problem):
    for part in [42, {'type': 'wheel'}, {'type': 'plate', 'w': 1.5}, {'type': 'imaginary'}]:
        ok, messages = explain(problem, [{'op': 'add', 'part': part}])
        assert not ok and messages


@pytest.mark.parametrize('body', [None, {}, [42]])
def test_malformed_input_build_fails_cleanly(problem, body):
    problem['full'] = body
    with pytest.raises(ValueError, match='input build'):
        apply(problem, [])
    ok, messages = explain(problem, [])
    assert not ok and 'input build' in messages[0]


def test_partial_step_cannot_pass_with_no_running_gear(problem):
    problem['full'] = [p for p in problem['full'] if p['type'] not in ('axle', 'wheel')]
    problem['edge'] = 1
    assert not explain(problem, [], partial=True)[0]


def test_functional_overlap_fails_the_full_candidate_check(problem):
    edits = [{'op': 'add', 'part': problem['full'][-1]}]
    assert not check(apply(problem, edits), problem['target'])[0]
    ok, messages = explain(problem, edits)
    assert not ok and any('overlap' in message for message in messages)


def test_partial_display_limit_cannot_hide_an_added_floating_part(problem):
    target = np.zeros((40, 5, 10), dtype=int)
    target[:8, :, :7] = problem['target']
    for x in range(20, 25):
        p = {'type': 'plate', 'w': 1, 'l': 1, 'x': x, 'y': 0, 'z': 6, 'color': 'red'}
        problem['full'].append(p)
        target[x, 0, 6] = COLOR_INDEX['red']
    problem.update(target=target, edge=1)
    added = {'type': 'plate', 'w': 1, 'l': 1, 'x': 35, 'y': 0, 'z': 6, 'color': 'red'}
    ok, messages = explain(problem, [{'op': 'add', 'part': added}], partial=True)
    assert not ok and any('your added' in message and 'loose' in message for message in messages)


def test_partial_step_requires_the_requested_line(problem):
    problem['edge'] = 4  # the input car's working axle lines are at 1 and 7
    assert not explain(problem, [], partial=True)[0]


def test_partial_step_with_one_completed_line_is_explicitly_partial(problem):
    problem['full'] = [p for p in problem['full'] if not (p['type'] in ('wheel', 'axle') and p['x'] == 7)]
    problem['edge'] = 1
    assert explain(problem, [], partial=True)[0]
    assert check(apply(problem, []), problem['target'], partial_edge=1)[0]
    assert not check(apply(problem, []), problem['target'])[0]


@pytest.mark.parametrize('text', ['{}', '[] []', '[{"op":"add"}] trailing text', '```json\n[]'])
def test_ambiguous_or_non_list_answers_are_rejected(text):
    with pytest.raises(ValueError):
        edits_from(text)
