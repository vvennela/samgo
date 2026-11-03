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
from .certify import certify
from .function_spec import MODEL as SPEC_MODEL
from .function_spec import infer
from .perceive import MAX_SIDE, SAM_3D_ENDPOINT, SAM_MODEL, perceive
from .running_gear_task import check
from .stability import stability
from .technic import structural, verify_function
from .verifier import MatchConfig, verify
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
    # Color is paint: shape and structure are checked, color accuracy is reported but never gates a build.
    config = MatchConfig(min_color_accuracy=0.0)
    structure = verify(solids, target, config=config, anchors=anchors)
    fidelity = verify(solids, original, config=config, anchors=anchors)
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


def _spark_reports(body, parts, report):
    """Final checks for an object whose mechanisms Spark named and Glimmer built: the same generic checker Glimmer
    was graded by (samgo.free_edit_test.check: real parts incl. Technic, connected through studs and joints, stands),
    asked for every mechanism that passed (rotors turn all the way round, freed surfaces swing)."""
    from .free_edit_test import check

    reqs = []
    for axis in ("horizontal", "vertical"):
        n = sum(1 for r in report.get("rotors", []) if r.get("passed") and r.get("axis", "horizontal") == axis)
        if n:
            reqs.append({"name": f"{axis}_rotors", "kind": "rotor", "axis": axis, "count": n})
        n = sum(1 for s in report.get("surfaces", []) if s.get("passed") and s.get("axis") == axis)
        if n:
            reqs.append({"name": f"{axis}_surfaces", "kind": "swing", "axis": axis, "degrees": 20, "count": n})
    index = {json.dumps(p, sort_keys=True): i for i, p in enumerate(body)}
    sources = [index.get(json.dumps(p, sort_keys=True), -1) for p in parts]  # recolored parts count as kept
    out = check(body, parts, sources, reqs, stand=True)
    ok = lambda k: bool(out[k][0]) if k in out else True
    reports = {'structure': {'passed': ok('structure') and ok('connected')},
               'function': {'passed': all(ok(q['name']) for q in reqs), 'requirements': reqs} if reqs
               else {'status': 'not_applicable'},
               'stability': {'passed': ok('stands')},
               'checks': {k: list(v) for k, v in out.items()}}
    return all(ok(k) for k in out), reports


def _door_reports(parts):
    """Final checks for a car with hinged doors: the door module's checks plus each door's own sampled swing."""
    from .doors import CAR_REQUIREMENTS, checks
    from .learned_mechanisms import car_door_context
    from .kinematics import bound_motion

    out = checks(parts, CAR_REQUIREMENTS, {}, vehicle=True)
    doors = {k: bound_motion(parts, g, d, 'vertical') for k, (g, d) in car_door_context(parts)['moving'].items()}
    ok = lambda v: v if isinstance(v, bool) else v[0]
    reports = {'structure': {'passed': all(ok(out[k]) for k in ('structure', 'connected') if k in out)},
               'function': {'passed': all(ok(v) for k, v in out.items() if k not in ('structure', 'connected', 'stands'))},
               'stability': {'passed': ok(out.get('stands', False))},
               'doors': {'passed': all(ok(v) for v in doors.values()) and bool(doors), 'each': {k: list(v) for k, v in doors.items()}},
               'checks': {k: list(v) if not isinstance(v, bool) else v for k, v in out.items()}}
    passed = all(ok(v) for v in out.values()) and reports['doors']['passed']
    return bool(passed), reports


def place_case_input(model: list[dict], family: str, ctx: dict) -> list[dict]:
    from .gen_objects import place_case
    return place_case(model, family, ctx, 'input')['input']


def _mechanism_reports(parts, family, ctx):
    """Final checks for a mechanism object: its family's native checks (structure, stability, every motion)."""
    from .audit_objects import native_checks
    from .learned_mechanisms import moving_groups

    ctx = dict(ctx)
    if 'moving' in ctx:  # recompute the moving groups on the final parts (their order may differ)
        kinds = {'tv': {'turntable_top'}, 'laptop': {'click_b'}}.get(family, {'hinge_top'})
        groups = moving_groups(parts, kinds)
        ctx['moving'] = {k: (groups[n], d) for n, (k, (_, d)) in enumerate(ctx['moving'].items()) if n < len(groups)}
    if 'seat' in ctx:
        i = max((i for i, p in enumerate(parts) if p['type'] == 'turntable_top'), default=None)
        ctx['seat'] = list(range(i, len(parts))) if i is not None else []
    out = native_checks({'family': family, 'mode': 'repair', 'context': ctx}, parts)
    ok = lambda v: v if isinstance(v, bool) else v[0]
    motion = {k: v for k, v in out.items() if k not in ('structure', 'connected', 'stands')}
    reports = {'structure': {'passed': all(ok(out[k]) for k in ('structure', 'connected') if k in out)},
               'function': {'passed': all(ok(v) for v in motion.values())},
               'stability': {'passed': ok(out.get('stands', True))},
               'checks': {k: list(v) if not isinstance(v, bool) else v for k, v in out.items()}}
    return all(ok(v) for v in out.values()), reports


def run(photo: Path, prompt: str, out: Path, car: bool | None = None, size: str = 'medium', reuse: bool = True,
        *, assembly: str = 'procedural', model: str | None = None, rounds: int = 3, max_tokens: int = 8000,
        timeout: float = 120, sequential: bool = True, mesh: Path | None = None,
        replay: Path | None = None, sampler=None, require_certificate: bool = False,
        certificate_timeout: float = 20, doors: bool = False, glass: bool = True, mechanisms: bool = True,
        layout_from_spark: bool = False,
        buildable: bool = False, manual: bool = False) -> dict:
    if assembly not in ('procedural', 'learned') or size not in SIZES:
        raise ValueError('unknown assembly mode or size')
    if assembly == 'procedural' and (model or replay or sampler is not None):
        raise ValueError('model, replay and sampler require learned assembly mode')
    if replay and (model or sampler is not None):
        raise ValueError('replay cannot be combined with a live model or sampler')
    config = Config(rounds, max_tokens, timeout, sequential)
    if not np.isfinite(certificate_timeout) or certificate_timeout <= 0:
        raise ValueError('certificate timeout must be positive and finite')
    photo_sha = file_hash(photo)
    out = out.resolve()
    run_dir = out / 'artifacts' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    start = time.monotonic()
    summary = {'photo': str(photo.resolve()), 'photo_sha256': photo_sha, 'prompt': prompt, 'size': size,
               'run_dir': str(run_dir), 'assembly_mode': assembly, 'passed': False, 'status': 'running',
               'require_certificate': require_certificate, 'certified_under_model': False,
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
        family, mechanism_ctx = None, None
        spark_report, spark_body = None, None
        if car:
            # The builder returns its last candidate even when no stable layout was found.
            # Validate scaffolding before spending model calls, with a small deterministic search.
            # Spark (the knowing model) reads the car's layout: wheel centres and the door outline on the model.
            layout = {}
            if layout_from_spark:
                try:
                    from .car_layout import layout as read_layout
                    oriented = original.transpose(1, 0, 2)[::-1].copy() if original.shape[1] > original.shape[0] \
                        else original
                    layout = read_layout(photo, oriented, run_dir / 'layout')
                except Exception as e:  # noqa: BLE001 - no Spark: the geometry-only rules apply
                    layout = {'error': f'{type(e).__name__}: {e}'[:200]}
                summary['layout'] = layout
            wheel_x = [layout['front_wheel_x'], layout['rear_wheel_x']] if 'front_wheel_x' in layout else None
            body_attempts = []
            for seed in range(3):
                reference = build_arched_car(original, attempts=6, seed=seed,
                                             **({'buildable': True} if buildable else {}),
                                             **({'wheel_x': wheel_x} if wheel_x else {}))
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
            # A known mechanism object: the research step says what it is and what must move; the matching builder
            # makes it, sized from the photo, and (learned mode) Glimmer places its moving-part connections.
            if mechanisms:
                stage('research')
                from .learned_mechanisms import family_of, model_for, place
                from .research import research
                key = digest({'photo': photo_sha, 'code': summary['code_sha256']['research.py']})
                path = cache / f'research-{key}.json'
                found = _cached_json(path) if reuse else None
                if found is None:
                    found = asyncio.run(async_stage(research(photo=photo), 180))
                    save(path, found)
                save(run_dir / 'research.json', found)
                family = family_of(found.get('object', ''), found.get('moving_parts'))
                summary.update(research_object=found.get('object'), mechanism_family=family,
                               research_requirements=found.get('requirements'))
            if family is None and mechanisms and assembly == 'learned':
                # Any other object: Spark names its parts and hinged surfaces and grounds each on the model, code
                # makes a buildable handoff, Glimmer builds it, code checks (samgo.spark_mechanisms). Nothing
                # Glimmer fails at is built by code instead.
                stage('spark_mechanisms')
                from .spark_mechanisms import build as spark_build
                chooser = sampler if sampler is not None else FireworksSampler(model)
                spark_body = parts
                parts, spark_report = spark_build(photo, target, parts, found.get('object') or prompt, chooser,
                                                  run_dir / 'spark_mechanisms', rounds=rounds)
                summary['spark_mechanisms'] = [{k: s.get(k) for k in ('part', 'name', 'handoff', 'passed', 'error')}
                                               for s in spark_report['surfaces']]
                summary['spark_unsupported'] = [(u['part'], u['name']) for u in spark_report['unsupported']]
                summary['spark_rotors'] = [{k: s.get(k) for k in ('part', 'name', 'site', 'passed', 'error')}
                                           for s in spark_report.get('rotors', [])]
                gear = spark_report.get('gear') or {}
                summary['spark_legs'] = [{k: g.get(k) for k in ('name', 'passed', 'error')}
                                         for g in gear.get('legs', [])]
                summary['spark_stands'] = gear.get('stands')
                summary['spark_legs_kept'] = gear.get('kept', True)
                glass_report = spark_report.get('glass') or {}
                summary['spark_glass'] = {k: glass_report.get(k) for k in ('boxes', 'cells', 'glass_parts', 'error')}
            if family is not None:
                stage('mechanism')
                reference_model, mechanism_ctx = model_for(family, original)
                save(run_dir / 'mechanism_reference.json', reference_model)
                if assembly == 'learned':
                    chooser = sampler if sampler is not None else FireworksSampler(model)
                    placed, mech_trace = place(reference_model, family, mechanism_ctx, chooser, run_dir / 'mechanism',
                                               rounds=rounds, max_tokens=max_tokens, timeout=timeout)
                    summary['mechanism_passed'] = mech_trace['passed']
                    summary['artifacts']['mechanism_trace'] = str(run_dir / 'mechanism' / 'trace.json')
                    # no procedural fallback: if Glimmer fails, the result is the object without its mechanism
                    parts = placed if placed is not None else place_case_input(reference_model, family, mechanism_ctx)
                else:
                    summary['mechanism_passed'] = True
                    parts = reference_model
        door_reports = None
        if car and doors:
            stage('doors')
            from .doors import add_car_doors
            try:
                spark = {}
                if car and (layout.get('doors') or layout.get('door')):  # every side door Spark saw, front to back
                    spark['outline'] = layout.get('doors') or [layout['door']]
                if car and 'front_wheel_x' in layout:  # which end is the front: doors hinge on their front edge
                    spark['front_high'] = layout['front_wheel_x'] > layout['rear_wheel_x']
                built = add_car_doors(parts, target, **spark)
            except ValueError as e:  # no room for doors on this body: the car is built without them
                built = {'parts': parts, 'checks': {'door_cut': (False, str(e))}}
            failed = {k: v for k, v in built['checks'].items() if not (v if isinstance(v, bool) else v[0])}
            summary['doors_cut'] = not failed
            if 'doors_seen' in built:  # e.g. four seen, two built: no room for the rear pair at this size
                summary.update(doors_seen=built['doors_seen'], doors_built=built['doors_built'])
            save(run_dir / 'doors_cut.json', built['parts'])
            if failed:
                summary['doors_passed'] = False
                summary['doors_error'] = 'door cut failed: ' + '; '.join(f'{k}: {v[1]}' for k, v in failed.items())
            elif assembly == 'learned':
                from .learned_mechanisms import car_door_context, place
                chooser = sampler if sampler is not None else FireworksSampler(model)
                # Only the hinges are left out: the running gear is already Glimmer's own, from the stage before.
                placed, door_trace = place(built['parts'], 'door', car_door_context(built['parts']), chooser,
                                           run_dir / 'doors', rounds=rounds, max_tokens=max_tokens, timeout=timeout,
                                           only={'hinge_base', 'hinge_top'})
                summary['doors_passed'] = door_trace['passed']
                summary['artifacts']['door_trace'] = str(run_dir / 'doors' / 'trace.json')
                if placed is not None:  # no procedural fallback: a failed placement leaves the car without doors
                    parts = placed
            else:
                summary['doors_passed'] = True
                parts = built['parts']
        before_glass = parts
        if car and glass:
            from .glass import add_glass
            parts, summary['glass_parts'] = add_glass(parts)
            summary['glass_cuts'] = len(parts) - len(before_glass)
        if car:  # paint: one body color (Spark's, else the photo's most common), black frame and underbody
            from .colors import body_color, paint
            from .parts import COLORS
            if spec is None:  # car given explicitly: still ask Spark for the paint color (cached per photo)
                key = digest({'photo': photo_sha, 'model': SPEC_MODEL, 'code': summary['code_sha256']['function_spec.py']})
                path = cache / f'spec-{key}.json'
                spec = _cached_json(path) if reuse else None
                if spec is None:
                    try:
                        spec = asyncio.run(async_stage(infer(photo), 120))
                        save(path, spec)
                    except Exception:  # noqa: BLE001 - no Spark: fall back to the photo's most common color
                        spec = {}
            chosen = (spec or {}).get('body_color')
            summary['body_color'] = chosen if chosen in COLORS else body_color(original)
            parts = paint(parts, summary['body_color'])
            # fewer, bigger pieces: neighbouring same-color plates in one rigid body become one real plate
            from .consolidate import consolidate
            parts, summary['merged_plates'] = consolidate(parts)
        from .colors import legalize
        parts, recolored = legalize(parts)
        summary['recolored_to_produced'] = len(recolored)
        certified = None
        if manual and car and glass and parts:
            # The manual is part of the verified result: if cutting plates at window edges leaves pieces no step
            # order can place, the windows are made of whole parts instead (fewer glass pieces, same shape).
            from .manual import certify_manual
            stage('manual')
            certified = certify_manual(parts)
            if not certified['passed'] and summary.get('glass_cuts'):
                from .colors import paint as repaint
                from .consolidate import consolidate as merge
                alt, alt_glass = add_glass(before_glass, cut=False)
                alt, alt_merged = merge(repaint(alt, summary['body_color']))
                alt, alt_recolored = legalize(alt)
                second = certify_manual(alt)
                summary['glass_mode'] = 'whole parts (manual)' if second['passed'] else 'cut at window edges'
                if second['passed']:  # the summary always describes the version that is kept
                    parts, certified, recolored = alt, second, alt_recolored
                    summary.update(glass_parts=alt_glass, merged_plates=alt_merged,
                                   recolored_to_produced=len(alt_recolored))
            else:
                summary['glass_mode'] = 'cut at window edges'
        save(run_dir / 'recolored.json', recolored)
        if any(p['type'].startswith('hinge') for p in parts) and car:
            door_reports = _door_reports(parts)
        elif family is not None:
            door_reports = _mechanism_reports(parts, family, mechanism_ctx)
        elif spark_report is not None:
            door_reports = _spark_reports(spark_body, parts, spark_report)
        np.savez_compressed(run_dir / 'assembly_target.npz', target=target)
        save(run_dir / 'build.json', parts)
        summary['artifacts']['build'] = str(run_dir / 'build.json')
        summary['parts'] = len(parts)
        summary['timings']['construction_total'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('validation')
        if door_reports is not None:
            passed, reports = door_reports
        else:
            passed, reports = _reports(parts, target, original, car)
        formal = certify(parts, timeout=certificate_timeout)
        save(run_dir / 'certificate.json', formal)
        summary['artifacts']['certificate'] = str(run_dir / 'certificate.json')
        summary['certified_under_model'] = formal['passed']
        reports['certification'] = {k: v for k, v in formal.items() if k != 'certificate'}
        reports['assembly_access'] = {
            'status': 'certified_vertical_envelope_path' if formal['passed'] else 'not_certified',
            'physical_validation': 'not_performed'}
        # A flag left None was not applicable to this object (no running gear, doors or family mechanism).
        summary['passed'] = passed and all(summary.get(k) is not False
                                           for k in ('assembly_passed', 'doors_passed', 'mechanism_passed'))
        if require_certificate:
            summary['passed'] = summary['passed'] and formal['passed']
        summary.update(structure_passed=reports['structure']['passed'],
                       function_passed=reports['function'].get('passed'), stands=reports['stability'].get('passed'))
        if 'doors' in reports:
            summary['doors_swing'] = reports['doors']['passed']
        save(run_dir / 'checks.json', reports)
        summary['artifacts']['checks'] = str(run_dir / 'checks.json')
        summary['timings']['validation'] = time.monotonic() - tick

        tick = time.monotonic()
        stage('exports')
        if parts:
            from .viewer3d import facts_for, page
            label = f"{assembly} / {summary.get('sampler', {}).get('source', 'procedural')} / " + (
                'checks passed' if summary['passed'] else 'incomplete: checks failed')
            label += ' / certified under static model' if formal['passed'] else ' / buildability not certified'
            (run_dir / 'viewer.html').write_text(page([(label, parts, facts_for(parts))]))
            summary['artifacts']['viewer'] = str(run_dir / 'viewer.html')
            from .ldraw import export as ldraw_export
            text, geometry = ldraw_export(parts, 'samgo_model')
            (run_dir / 'model.ldr').write_text(text)
            summary['artifacts']['ldraw'] = str(run_dir / 'model.ldr')
            summary['ldraw_worst_mismatch_mm'] = max((r['mismatch_mm'] for r in geometry), default=0)
        if summary['passed']:
            from .instructions import write
            options = {'certificate': formal['certificate']} if formal['passed'] else {}
            summary['artifacts']['instructions'] = str(write(parts, run_dir / 'instructions', **options))
        if manual and parts:
            from .manual import certify_manual
            if certified is None:
                certified = certify_manual(parts)
            save(run_dir / 'manual.json', certified)
            summary['artifacts']['manual'] = str(run_dir / 'manual.json')
            summary.update(manual_certified=certified['passed'], manual_steps=len(certified['steps']),
                           manual_orientation=certified.get('orientation'))
        summary['timings']['exports'] = time.monotonic() - tick
        summary['status'] = 'passed' if summary['passed'] else 'failed'
        stage('complete')
    except Exception as e:
        # Leave a failed run manifest even for setup/API/export failures. No success artifacts are reused.
        import traceback
        traceback.print_exc()  # to the local log only: the summary never carries service details
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
    ap.add_argument('--doors', action='store_true', help='cut a hinged door in each side; with --assembly learned, Glimmer places the hinges')
    ap.add_argument('--no-glass', action='store_true', help='keep the cabin walls opaque')
    ap.add_argument('--no-spark-layout', action='store_true', help='cars: geometry only, no wheel/door layout from Spark')
    ap.add_argument('--no-mechanisms', action='store_true', help='static objects: skip the research step')
    ap.add_argument('--buildable', action='store_true', help='cars: lay out for structure first, then color (certifiable)')
    ap.add_argument('--manual', action='store_true', help='also certify the building instructions (samgo.manual)')
    ap.add_argument('--require-certificate', action='store_true', help='require the exact static assembly certificate; unsupported mechanisms fail closed')
    ap.add_argument('--certificate-timeout', type=float, default=20, help='solver budget for static certification')
    args = ap.parse_args()
    result = run(args.photo, args.prompt, args.out, args.car, args.size, reuse=not args.fresh,
                 assembly=args.assembly, model=args.model, replay=args.replay, mesh=args.mesh,
                 rounds=args.rounds, max_tokens=args.max_tokens, timeout=args.timeout, sequential=not args.joint,
                 require_certificate=args.require_certificate, certificate_timeout=args.certificate_timeout,
                 doors=args.doors, glass=not args.no_glass, layout_from_spark=not args.no_spark_layout, mechanisms=not args.no_mechanisms,
                 buildable=args.buildable, manual=args.manual)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
