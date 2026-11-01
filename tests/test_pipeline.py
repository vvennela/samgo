import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from samgo import pipeline
from tests.test_assembly import ScriptedSampler, answers

pytest_plugins = ['tests.test_assembly']
pytestmark = pytest.mark.filterwarnings('error::RuntimeWarning')


@pytest.fixture
def setup_pipeline(monkeypatch, tmp_path, reference):
    parts, target = reference
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(b'photo bytes')
    mesh = tmp_path / 'mesh.glb'
    mesh.write_bytes(b'mesh bytes')
    calls = []

    async def perceive(photo, prompt, out):
        calls.append((photo.read_bytes(), prompt))
        (out / 'object.glb').write_bytes(b'generated mesh' + photo.read_bytes() + prompt.encode())

    monkeypatch.setattr(pipeline, 'perceive', perceive)
    monkeypatch.setattr(pipeline, 'load_mesh', lambda path: path)
    monkeypatch.setattr(pipeline, 'voxelize', lambda mesh, size: target.copy())
    monkeypatch.setattr(pipeline, 'render', lambda target, out: out.write_bytes(b'preview'))
    monkeypatch.setattr(pipeline, 'build_arched_car', lambda target, attempts, seed: SimpleNamespace(
        parts=parts, target=target, wheels=[SimpleNamespace(edge=x, size='30.4x14') for x in (1, 7)], symmetric=1))
    return photo, mesh, tmp_path / 'out', calls, parts


def test_learned_pipeline_exports_the_checked_model_and_replays(setup_pipeline):
    photo, mesh, out, _, parts = setup_pipeline
    result = pipeline.run(photo, 'car', out, car=True, assembly='learned', mesh=mesh,
                          sampler=ScriptedSampler(answers(parts)))
    assert result['passed'] and result['stands'] and result['function_passed']
    assert result['perception']['source'] == 'supplied_mesh'
    assert not result['procedural_fallback']
    final = json.loads(Path(result['artifacts']['build']).read_text())
    trace = json.loads(Path(result['artifacts']['trace']).read_text())
    # Colors may change after assembly (produced colors, glass); geometry is exactly the accepted assembly's, and
    # the final checks ran on the saved, recolored model.
    strip = lambda parts: [{k: v for k, v in p.items() if k != 'color'} for p in parts]
    assert strip(final) == strip(trace['steps'][-1]['accepted'])
    assert 'checks passed' in Path(result['artifacts']['viewer']).read_text()
    assert Path(result['artifacts']['instructions']).is_file()
    replay = pipeline.run(photo, 'car', out, car=True, assembly='learned', mesh=mesh,
                          replay=Path(result['artifacts']['trace']))
    assert replay['passed'] and replay['sampler']['source'] == 'replay'
    assert replay['run_dir'] != result['run_dir']
    assert json.loads(Path(replay['artifacts']['build']).read_text()) == final


def test_failed_run_does_not_publish_old_success_or_instructions(setup_pipeline):
    photo, mesh, out, _, parts = setup_pipeline
    good = pipeline.run(photo, 'car', out, car=True, assembly='learned', mesh=mesh,
                        sampler=ScriptedSampler(answers(parts)))
    bad = pipeline.run(photo, 'car', out, car=True, assembly='learned', mesh=mesh, rounds=1,
                       sampler=ScriptedSampler(['[]']))
    assert good['passed'] and not bad['passed'] and bad['status'] == 'failed'
    assert 'instructions' not in bad['artifacts']
    assert 'incomplete: checks failed' in Path(bad['artifacts']['viewer']).read_text()
    assert json.loads((out / 'summary.json').read_text()) == bad
    assert bad['run_dir'] != good['run_dir']


def test_perception_cache_checks_photo_prompt_mesh_hash_and_fresh(setup_pipeline, monkeypatch):
    photo, _, out, calls, _ = setup_pipeline
    # Cache tests exercise perception and validation without spending time rendering duplicate instructions.
    import samgo.instructions
    monkeypatch.setattr(samgo.instructions, 'write', lambda parts, out: out / 'instructions.md')
    first = pipeline.run(photo, 'car', out, car=True)
    second = pipeline.run(photo, 'car', out, car=True)
    assert first['passed'] and second['passed']
    assert first['perception']['source'] == 'live' and second['perception']['source'] == 'cache'
    assert len(calls) == 1
    Path(first['run_dir'], 'object.glb').write_bytes(b'corrupted')
    assert pipeline.run(photo, 'car', out, car=True)['perception']['source'] == 'live'
    assert pipeline.run(photo, 'truck', out, car=True)['perception']['source'] == 'live'
    photo.write_bytes(b'different photo')
    assert pipeline.run(photo, 'car', out, car=True)['perception']['source'] == 'live'
    assert pipeline.run(photo, 'car', out, car=True, reuse=False)['perception']['source'] == 'live'
    assert len(calls) == 5


def test_instability_prevents_success_and_instruction_export(setup_pipeline, monkeypatch):
    from samgo.verifier import Issue, Report
    photo, mesh, out, _, _ = setup_pipeline
    monkeypatch.setattr(pipeline, 'stability', lambda *a, **k: Report(False, [Issue('unbalanced', 'tips')], {}))
    result = pipeline.run(photo, 'car', out, car=True, mesh=mesh)
    assert not result['passed'] and result['stands'] is False
    assert 'instructions' not in result['artifacts']


def test_perception_failure_leaves_failed_manifest(setup_pipeline, monkeypatch):
    photo, _, out, _, _ = setup_pipeline

    async def fail(*a):
        raise RuntimeError('private service details')
    monkeypatch.setattr(pipeline, 'perceive', fail)
    result = pipeline.run(photo, 'car', out, car=True)
    assert not result['passed'] and result['error_kind'] == 'RuntimeError'
    assert not result['artifacts']
    assert 'private service' not in (out / 'summary.json').read_text()


def test_learned_static_is_explicitly_unsupported(setup_pipeline):
    photo, _, out, _, _ = setup_pipeline
    with pytest.raises(ValueError, match='cars only'):
        pipeline.run(photo, 'chair', out, car=False, assembly='learned')


def test_body_search_rejects_bad_scaffolding_before_sampling(setup_pipeline, monkeypatch):
    photo, mesh, out, _, parts = setup_pipeline
    original = pipeline.build_arched_car
    seeds = []

    def candidate(target, attempts, seed):
        seeds.append(seed)
        return None if seed < 2 else original(target, attempts, seed)
    monkeypatch.setattr(pipeline, 'build_arched_car', candidate)
    sampler = ScriptedSampler(answers(parts))
    result = pipeline.run(photo, 'car', out, car=True, mesh=mesh, assembly='learned', sampler=sampler)
    assert result['passed'] and seeds == [0, 1, 2]
    assert [r['passed'] for r in result['body_search']] == [False, False, True]
    assert len(sampler.messages) == 2


def test_body_search_exhaustion_is_not_a_model_failure(setup_pipeline, monkeypatch):
    photo, mesh, out, _, _ = setup_pipeline
    monkeypatch.setattr(pipeline, 'build_arched_car', lambda *a, **k: None)
    sampler = ScriptedSampler([])
    result = pipeline.run(photo, 'car', out, car=True, mesh=mesh, assembly='learned', sampler=sampler)
    assert not result['passed'] and result['stage'] == 'body_construction'
    assert len(result['body_search']) == 3 and not sampler.messages
    assert 'instructions' not in result['artifacts']


def test_static_path_remains_available(setup_pipeline, monkeypatch):
    import numpy as np
    photo, mesh, out, _, _ = setup_pipeline
    target = np.ones((2, 2, 1), dtype=int)
    monkeypatch.setattr(pipeline, 'voxelize', lambda *a: target)
    result = pipeline.run(photo, 'block', out, car=False, mesh=mesh)
    assert result['passed'] and result['stands']
    assert result['function_passed'] is None
    assert 'trace' not in result['artifacts']
    assert result['certified_under_model']


def test_strict_certificate_gate_rejects_unsupported_vehicle(setup_pipeline):
    photo, mesh, out, _, _ = setup_pipeline
    result = pipeline.run(photo, 'car', out, car=True, mesh=mesh, require_certificate=True)
    assert not result['passed'] and not result['certified_under_model']
    assert 'instructions' not in result['artifacts']
    report = json.loads(Path(result['artifacts']['certificate']).read_text())
    assert report['status'] == 'unsupported' and report['physical_validation'] == 'not_performed'
    assert 'buildability not certified' in Path(result['artifacts']['viewer']).read_text()


def test_strict_certificate_gate_accepts_supported_static_build(setup_pipeline, monkeypatch):
    import numpy as np
    photo, mesh, out, _, _ = setup_pipeline
    monkeypatch.setattr(pipeline, 'voxelize', lambda *a: np.ones((2, 2, 1), dtype=int))
    result = pipeline.run(photo, 'block', out, car=False, mesh=mesh, require_certificate=True)
    assert result['passed'] and result['certified_under_model']
    assert 'gravity equilibrium' in Path(result['artifacts']['instructions']).read_text()


def test_export_error_reports_failure_in_its_own_run(setup_pipeline, monkeypatch):
    import samgo.instructions
    photo, mesh, out, _, _ = setup_pipeline

    def fail(*a):
        raise ValueError('unsupported instruction part')
    monkeypatch.setattr(samgo.instructions, 'write', fail)
    result = pipeline.run(photo, 'car', out, car=True, mesh=mesh)
    assert not result['passed'] and result['stage'] == 'exports'
    assert result['error'] == 'unsupported instruction part'
    assert 'instructions' not in result['artifacts']
