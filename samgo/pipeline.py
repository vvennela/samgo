"""Photo -> procedural body -> optional learned running gear -> checked artifacts.

    python -m samgo.pipeline photos/camry.jpg --prompt car --car --assembly learned -o runs/car

Each invocation gets a separate artifact directory. Explicit mesh inputs and model-answer
replays are labeled as such; perception reuse requires matching content/configuration hashes.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .arches import build_arched_car
from .assembly import Config, FireworksSampler, ReplaySampler, assemble, digest, save
from .builder import build
from .function_spec import MODEL as SPEC_MODEL
from .function_spec import infer
from .perceive import MAX_SIDE, SAM_3D_ENDPOINT, SAM_MODEL, perceive
from .running_gear_task import check
from .stability import stability
from .technic import structural, verify_function
from .verifier import verify
from .voxelize import SIZES, load_mesh, render, voxelize


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def async_stage(operation, timeout):
    return await asyncio.wait_for(operation, timeout=timeout)


def _cached_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _reports(parts, target, original, vehicle):
    solids = structural(parts) if vehicle else parts
    anchors = frozenset(i for i, p in enumerate(p for p in parts if p['type'] in ('plate', 'brick', 'technic_brick'))
                        if p['type'] == 'technic_brick') if vehicle else frozenset()
    structure = verify(solids, target, anchors=anchors)
    fidelity = verify(solids, original, anchors=anchors)
    function = verify_function(parts) if vehicle else None
    # Do not feed invalid geometry to the force solver or report a deferred check as passing.
    stands = stability(parts, vehicle=vehicle) if structure.passed and (function is None or function.passed) else None
    reports = {'structure': asdict(structure),
               'function': asdict(function) if function else {'status': 'not_applicable'},
               'stability': asdict(stands) if stands else {'status': 'not_evaluated'},
               'original_target_fidelity': fidelity.metrics,
               'original_target_issues': sorted(fidelity.codes),
               'assembly_access': {'status': 'not_evaluated'}}
    passed = structure.passed and (function is None or function.passed) and stands is not None and stands.passed
    if vehicle:
        canonical, issues = check(parts, target)
        reports['aggregate'] = {'passed': canonical, 'issues': issues}
        passed = passed and canonical
    return bool(passed), reports


def run(photo: Path, prompt: str, out: Path, car: bool | None = None, size: str = 'medium', reuse: bool = True,
        *, assembly: str = 'procedural', model: str | None = None, rounds: int = 3, max_tokens: int = 8000,
        timeout: float = 120, sequential: bool = True, mesh: Path | None = None,
        replay: Path | None = None, sampler=None) -> dict:
    if assembly not in ('procedural', 'learned') or size not in SIZES:
        raise ValueError('unknown assembly mode or size')
    if assembly == 'procedural' and (model or replay or sampler is not None):
        raise ValueError('model, replay and sampler require learned assembly mode')
    if replay and (model or sampler is not None):
        raise ValueError('replay cannot be combined with a live model or sampler')
    if assembly == 'learned' and car is False:
        raise ValueError('learned assembly currently supports cars only')
    config = Config(rounds, max_tokens, timeout, sequential)
    photo_sha = file_hash(photo)
    out = out.resolve()
    run_dir = out / 'artifacts' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    start = time.monotonic()
    summary = {'photo': str(photo.resolve()), 'photo_sha256': photo_sha, 'prompt': prompt, 'size': size,
               'run_dir': str(run_dir), 'assembly_mode': assembly, 'passed': False, 'status': 'running',
               'procedural_fallback': False, 'timings': {}, 'artifacts': {},
               'code_sha256': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')
                               if p.name != 'ldraw.py'}}
    summary['dependencies'] = {name: importlib.metadata.version(name) for name in
                               ('numpy', 'scipy', 'trimesh', 'matplotlib', 'meta-sam-parser', 'fal-client')}

    def finish():
        summary['timings']['total'] = time.monotonic() - start
        save(run_dir / 'summary.json', summary)
        save(out / 'summary.json', summary)  # Latest-run pointer; artifacts remain isolated per invocation.
        return summary

    def stage(name):
        summary['stage'] = name
        finish()

    finish()
    try:
        stage('function_spec')
        tick = time.monotonic()
        spec = None
        if car is None:
            key = digest({'photo': photo_sha, 'model': SPEC_MODEL, 'code': summary['code_sha256']['function_spec.py']})
            path = cache / f'spec-{key}.json'
            spec = _cached_json(path) if reuse else None
            summary['spec_source'] = 'cache' if spec is not None else 'live'
            if spec is None:
                spec = asyncio.run(async_stage(infer(photo), 120))
                save(path, spec)
            car = spec.get('kind') == 'vehicle'
        else:
            summary['spec_source'] = 'explicit_vehicle' if car else 'explicit_static'
        summary.update(function_spec=spec, vehicle=car)
        save(run_dir / 'function_spec.json', {'source': summary['spec_source'], 'spec': spec, 'vehicle': car})
        if assembly == 'learned' and not car:
            raise ValueError('learned assembly currently supports cars only')
        summary['timings']['function_spec'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('perception')
        dest = run_dir / 'object.glb'
        if mesh is not None:
            shutil.copyfile(mesh, dest)
            summary['perception'] = {'source': 'supplied_mesh', 'path': str(mesh.resolve()),
                                     'photo_mesh_association': 'user_supplied_not_verified'}
        else:
            key = digest({'photo': photo_sha, 'prompt': prompt, 'sam': SAM_MODEL, 'sam3d': SAM_3D_ENDPOINT,
                          'max_side': MAX_SIDE, 'code': summary['code_sha256']['perceive.py']})
            index = cache / f'perception-{key}.json'
            entry = _cached_json(index) if reuse else None
            cached = Path(entry['mesh']) if entry and 'mesh' in entry else None
            if cached and cached.is_file() and file_hash(cached) == entry.get('sha256'):
                shutil.copyfile(cached, dest)
                summary['perception'] = {'source': 'cache', 'key': key, 'origin': str(cached)}
            else:
                asyncio.run(async_stage(perceive(photo, prompt, run_dir), 300))
                save(index, {'mesh': str(dest), 'sha256': file_hash(dest)})
                summary['perception'] = {'source': 'live', 'key': key}
        summary['perception']['mesh_sha256'] = file_hash(dest)
        summary['timings']['perception'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('body_construction')
        original = voxelize(load_mesh(dest), SIZES[size])
        np.savez_compressed(run_dir / 'target.npz', target=original)
        render(original, run_dir / 'target.png')
        summary.update(target_shape=list(original.shape), cells=int((original > 0).sum()))
        target = original
        if car:
            # The builder returns its last candidate even when no stable layout was found.
            # Validate scaffolding before spending model calls, with a small deterministic search.
            body_attempts = []
            for seed in range(3):
                reference = build_arched_car(original, attempts=6, seed=seed)
                ok, issues = check(reference.parts, reference.target) if reference else (False, ['no supported body'])
                body_attempts.append({'seed': seed, 'attempts': 6, 'passed': ok, 'issues': issues})
                save(run_dir / 'body_search.json', body_attempts)
                if ok:
                    break
            summary['body_search'] = body_attempts
            if not ok:
                raise ValueError('no verified procedural body after 3 seeds: ' + '; '.join(issues))
            target = reference.target
            summary.update(wheels=[[w.edge, w.size] for w in reference.wheels], symmetric=round(reference.symmetric, 2))
            if assembly == 'learned':
                stage('learned_assembly')
                selected = sampler if sampler is not None else ReplaySampler(replay) if replay else FireworksSampler(model)
                summary['timings']['body_construction'] = time.monotonic() - tick
                parts, trace = assemble(reference.parts, target, run_dir / 'assembly', selected, config)
                summary.update(assembly_passed=trace['passed'], sampler=trace['sampler'])
                summary['timings']['learned_assembly'] = trace['seconds']
                summary['artifacts']['trace'] = str(run_dir / 'assembly' / 'trace.json')
            else:
                parts = reference.parts
        else:
            parts = build(target).pieces
        np.savez_compressed(run_dir / 'assembly_target.npz', target=target)
        save(run_dir / 'build.json', parts)
        summary['artifacts']['build'] = str(run_dir / 'build.json')
        summary['parts'] = len(parts)
        summary['timings']['construction_total'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('validation')
        passed, reports = _reports(parts, target, original, car)
        summary['passed'] = passed and summary.get('assembly_passed', True)
        summary.update(structure_passed=reports['structure']['passed'],
                       function_passed=reports['function'].get('passed'), stands=reports['stability'].get('passed'))
        save(run_dir / 'checks.json', reports)
        summary['artifacts']['checks'] = str(run_dir / 'checks.json')
        summary['timings']['validation'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('exports')
        if parts:
            from .viewer3d import facts_for, page
            label = f"{assembly} / {summary.get('sampler', {}).get('source', 'procedural')} / " + (
                'checks passed' if summary['passed'] else 'incomplete: checks failed')
            (run_dir / 'viewer.html').write_text(page([(label, parts, facts_for(parts))], simulate=False))
            summary['artifacts']['viewer'] = str(run_dir / 'viewer.html')
        if summary['passed']:
            from .instructions import write
            summary['artifacts']['instructions'] = str(write(parts, run_dir / 'instructions'))
        summary['timings']['exports'] = time.monotonic() - tick
        summary['status'] = 'passed' if summary['passed'] else 'failed'
        stage('complete')
    except Exception as e:
        # Leave a failed run manifest even for setup/API/export failures. No success artifacts are reused.
        summary.update(passed=False, status='failed', error_kind=type(e).__name__)
        summary['error'] = str(e) if isinstance(e, (ValueError, FileNotFoundError, ImportError)) else 'stage failed; inspect local runtime'
    return finish()


def main() -> None:
    ap = argparse.ArgumentParser(description='photo -> assembly -> checked viewer and instructions')
    ap.add_argument('photo', type=Path)
    ap.add_argument('--prompt', required=True)
    kind = ap.add_mutually_exclusive_group()
    kind.add_argument('--car', dest='car', action='store_const', const=True)
    kind.add_argument('--static', dest='car', action='store_const', const=False)
    ap.add_argument('--size', choices=SIZES, default='medium')
    ap.add_argument('-o', '--out', type=Path, required=True)
    ap.add_argument('--fresh', action='store_true', help='bypass perception and function-spec caches')
    ap.add_argument('--assembly', choices=('procedural', 'learned'), default='procedural')
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--model', help='Fireworks sampler checkpoint; default is the base model')
    source.add_argument('--replay', type=Path, help='exact-prompt replay of a recorded assembly trace')
    ap.add_argument('--mesh', type=Path, help='explicit precomputed mesh; skips fresh perception')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=8000)
    ap.add_argument('--timeout', type=float, default=120, help='seconds to wait for each sampling response')
    ap.add_argument('--joint', action='store_true', help='solve both axle lines in one task')
    args = ap.parse_args()
    result = run(args.photo, args.prompt, args.out, args.car, args.size, reuse=not args.fresh,
                 assembly=args.assembly, model=args.model, replay=args.replay, mesh=args.mesh,
                 rounds=args.rounds, max_tokens=args.max_tokens, timeout=args.timeout, sequential=not args.joint)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
