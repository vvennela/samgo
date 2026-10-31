"""Learned running-gear orchestration shared by the photo pipeline and demo.

The sampler receives only model-visible messages. Procedural reference answers are never sent.
Traces can be replayed only against the exact recorded messages, without an inference service.
"""

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .running_gear_task import (FEEDBACK, apply, axles_of, check, edits_from, explain, frame_problem, pose_frame,
                                prompt, step_problem)

MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"
VERSION = "running-gear-assembly-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path: Path, value) -> None:
    """Replace a JSON report atomically, including progress before sampling."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@dataclass(frozen=True)
class Config:
    rounds: int = 3
    max_tokens: int = 8000
    timeout: float = 120
    sequential: bool = True

    def __post_init__(self):
        if self.rounds < 1 or self.max_tokens < 1 or not 0 < self.timeout < float('inf'):
            raise ValueError('rounds, max_tokens and timeout must be positive and finite')


@dataclass(frozen=True)
class Sample:
    text: str
    tokens: int | None = None


class FireworksSampler:
    """Lazy adapter for the same external training runtime used by the original demo."""

    def __init__(self, model: str | None = None):
        self.metadata = {'source': 'live', 'model': model or MODEL, 'tokenizer': HF_MODEL}
        self.model = model
        self.client = None

    def __call__(self, messages, *, max_tokens, temperature, timeout):
        if self.client is None:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / '.env')
            from fireworks.training.sdk import FiretitanServiceClient
            from training.utils.serverless import serverless_base_url
            from training.utils.supervised import build_renderer
            from training.utils.tokenizers import load_tokenizer

            tok = load_tokenizer(HF_MODEL)
            self.renderer = build_renderer(tok, HF_MODEL, 'muse_glimmer')
            service = FiretitanServiceClient(api_key=os.environ['FIREWORKS_API_KEY'],
                                            base_url=serverless_base_url('https://api.fireworks.ai'))
            self.client = service.create_sampling_client(tokenizer=tok, **(
                {'model_path': self.model} if self.model else {'base_model': MODEL}))
        from fireworks.training import sdk
        from training.renderer import get_text_content

        params = sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                    stop=self.renderer.get_stop_sequences())
        tokens = list(self.client.sample(prompt=self.renderer.build_generation_prompt(messages), num_samples=1,
                                         sampling_params=params).result(timeout=timeout).sequences[0].tokens)
        message, _ = self.renderer.parse_response(tokens)
        return Sample(get_text_content(message), len(tokens))


class ReplaySampler:
    """Replay recorded text; fail closed on a different prompt or exhausted recording."""

    def __init__(self, path: Path):
        raw = path.read_bytes()
        trace = json.loads(raw)
        self.reference_hash = trace['reference_sha256']
        self.target_hash = trace['target_sha256']
        self.target_shape = trace.get('target_shape')
        self.target_dtype = trace.get('target_dtype')
        self.records = iter(attempt for step in trace['steps'] for attempt in step['attempts'])
        self.metadata = {'source': 'replay', 'trace': str(path.resolve()),
                         'sha256': hashlib.sha256(raw).hexdigest(), 'recorded_sampler': trace['sampler']}

    def validate(self, reference, target):
        if (digest(reference) != self.reference_hash or hashlib.sha256(target.tobytes()).hexdigest() != self.target_hash
                or self.target_shape is not None and list(target.shape) != self.target_shape
                or self.target_dtype is not None and str(target.dtype) != self.target_dtype):
            raise ValueError('replay assembly/target mismatch; recording belongs to a different input')

    def __call__(self, messages, **kwargs):
        try:
            attempt = next(self.records)
        except StopIteration as e:
            raise ValueError('replay has no remaining answer') from e
        if attempt['messages'] != messages:
            raise ValueError('replay prompt mismatch; this recording belongs to a different task/state')
        if attempt.get('raw_answer') is None:
            raise ValueError('recorded attempt has no model answer')
        return Sample(attempt['raw_answer'], attempt.get('tokens'))


def assemble(reference: list[dict], target, out: Path, sampler, config: Config = Config()) -> tuple[list[dict], dict]:
    """Only accepted edits advance the build; failure returns the last accepted state, never a fallback."""
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trace = {'version': VERSION, 'sampler': sampler.metadata, 'config': asdict(config), 'steps': [],
             'status': 'running', 'passed': False, 'reference_sha256': digest(reference),
             'target_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
             'target_shape': list(target.shape), 'target_dtype': str(target.dtype),
             'scaffolding': 'procedural body, tire regions and wheel sizes; model adds beams, axles, wheels and bushes'}
    trace_path = out / 'trace.json'
    save(out / 'reference.json', reference)
    np.savez_compressed(out / 'target.npz', target=target)
    trace['code_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('assembly.py', 'running_gear_task.py', 'edits.py', 'technic.py',
                                         'verifier.py', 'stability.py', 'parts.py')}
    save(trace_path, trace)
    if isinstance(sampler, ReplaySampler):
        try:
            sampler.validate(reference, target)
        except ValueError as e:
            trace.update(status='failed', error=str(e), seconds=time.monotonic() - started)
            save(trace_path, trace)
            return [], trace
    edges = axles_of(reference)
    problem = pose_frame(reference, target, edges)
    if problem is None:
        trace.update(status='failed', error='could not pose a verified frame task', seconds=time.monotonic() - started)
        save(trace_path, trace)
        return [], trace
    build = copy.deepcopy(problem['full'])
    save(out / 'input.json', build)
    save(out / 'accepted.json', build)
    sizes = {p['x']: p['size'] for p in reference if p['type'] == 'wheel'}
    sequential = config.sequential
    if sequential and frame_problem(reference, target, 'front_first')[0] is None:
        # A small shell can miss the full-target IoU threshold while its other axle is absent.
        # Use the already verified joint task; never ask the model to solve an infeasible partial task.
        sequential = False
        trace['task_mode_reason'] = 'partial reference fails checks; using the verified joint task'
    trace['task_mode'] = 'sequential' if sequential else 'joint'
    tasks = edges if sequential else [None]
    all_steps = True
    for k, edge in enumerate(tasks):
        partial = sequential and k < len(tasks) - 1
        prob = step_problem(build, target, edge, sizes[edge], frame=True) if sequential else problem
        chat = prompt(prob)
        step = {'edge': edge, 'partial': partial, 'input': copy.deepcopy(build), 'local': prob['local'],
                'tires': prob['tires'], 'attempts': [], 'passed': False}
        trace['steps'].append(step)
        for rnd in range(1, config.rounds + 1):
            attempt = {'round': rnd, 'messages': copy.deepcopy(chat), 'temperature': 0.0 if rnd == 1 else 0.7,
                       'raw_answer': None, 'tokens': None, 'edits': None, 'candidate': None, 'passed': False}
            step['attempts'].append(attempt)
            save(trace_path, trace)
            tick = time.monotonic()
            try:
                sample = sampler(copy.deepcopy(chat), max_tokens=config.max_tokens,
                                 temperature=attempt['temperature'], timeout=config.timeout)
                attempt.update(raw_answer=sample.text, tokens=sample.tokens,
                               at_token_limit=sample.tokens is not None and sample.tokens >= config.max_tokens)
                chat.append({'role': 'assistant', 'content': sample.text})
            except Exception as e:  # service/runtime failures are recorded without exposing credential-bearing messages
                message = (str(e) if isinstance(sampler, ReplaySampler) else
                           'sampling runtime unavailable: install the Fireworks training SDK and expose the training cookbook'
                           if isinstance(e, ImportError) else 'sampling failed; check runtime, credentials or timeout')
                attempt.update(error_kind=type(e).__name__, issues=[message])
                attempt['seconds'] = time.monotonic() - tick
                save(trace_path, trace)
                continue
            try:
                edits = edits_from(sample.text)
                attempt['edits'] = edits
                candidate = apply(prob, edits)
                attempt['candidate'] = candidate
                ok, issues = explain(prob, edits, partial=partial)
                attempt.update(passed=ok, issues=issues)
            except (ValueError, TypeError, KeyError) as e:
                ok, issues = False, [f'invalid answer: {e}']
                attempt.update(error_kind=type(e).__name__, issues=issues)
            attempt['seconds'] = time.monotonic() - tick
            if ok:
                build = candidate
                step.update(passed=True, accepted=copy.deepcopy(build))
                save(out / 'accepted.json', build)
                save(trace_path, trace)
                break
            chat.append({'role': 'user', 'content': FEEDBACK.format(issues='\n'.join(f'- {i}' for i in issues))})
            save(trace_path, trace)
        if not step['passed']:
            all_steps = False
            break
    final_ok, issues = check(build, target)
    trace.update(passed=all_steps and final_ok, status='passed' if all_steps and final_ok else 'failed',
                 final_issues=issues, seconds=time.monotonic() - started,
                 output_tokens=sum(a['tokens'] or 0 for s in trace['steps'] for a in s['attempts']))
    trace['live_output_tokens'] = trace['output_tokens'] if sampler.metadata.get('source') == 'live' else 0
    save(trace_path, trace)
    return build, trace
