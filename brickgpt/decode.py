"""BrickGPT inference: the model places one brick at a time; bad bricks are resampled, unstable builds roll back.

Rules and defaults follow BrickGPT's released code (models/brickgpt.py): temperature 0.6, +0.01 per rejected
attempt up to 2.0, top-k 20, at most 500 rejections per brick, 2000 bricks and 100 rollbacks. Each step asks
an OpenAI-compatible /v1/completions server (Fireworks, vLLM, llama.cpp) for one more line, with the prompt
rendered from Glimmer's chat template and the answer pre-filled up to the bricks placed so far, so the
server's prompt cache reuses the growing build. Every accept, reject and rollback is logged as an event.

    uv run python -m brickgpt.decode "a low wooden table with four legs" --base-url URL --model MODEL
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import messages
from .structure import Structure, parse

ANSWER_PREFIX = "<|start|>assistant to=user<|message|>"   # how a training answer begins (no reasoning)
END = "<|eot|>"


@dataclass
class Config:
    temperature: float = 0.6
    temperature_increase: float = 0.01
    max_temperature: float = 2.0
    top_k: int = 20
    max_brick_rejections: int = 500
    max_bricks: int = 2000
    max_regenerations: int = 100
    max_line_tokens: int = 16
    # Lines asked for per model call. Lines are accepted in order up to the first bad one, then the rest is
    # dropped and sampling resumes from there: the accepted bricks are distributed exactly as with one call
    # per brick (generation is left to right anyway), with several times fewer calls.
    lines_per_call: int = 8
    # Best-of-k per part (decoder.md option 3): ask for k single-line candidates and keep the one the structure
    # scores highest (ShapeChecked.score: still-empty target cells filled). 1 = off.
    best_of: int = 1
    # Times the model's end of answer is refused (structure.allow_end() is False) before it is accepted.
    max_end_refusals: int = 30
    # A line already rejected at this position is a repeat: it raises the temperature this much more, and after
    # max_repeats repeats at one position the model counts as stuck and the build stops there.
    repeat_temperature_jump: float = 0.05
    max_repeats: int = 10
    # When stuck, the parts just placed usually set up the pattern the model is locked into (a column it keeps
    # extending): remove the last `backtrack_parts` and let it continue from there, at most `max_backtracks` times.
    # Stuck again without getting past the last stuck point doubles the amount removed (2, 4, 8, ...).
    backtrack_parts: int = 2
    max_backtracks: int = 5


@dataclass
class Result:
    structure: Structure
    stable: bool
    events: list = field(default_factory=list)
    calls: int = 0
    rejections: int = 0
    rollbacks: int = 0
    seconds: float = 0.0
    gave_up: str = ""


def render_prompt(tokenizer, caption: str) -> str:
    """Chat-template text up to the start of the answer, exactly as the training rows render."""
    text = tokenizer.apply_chat_template(messages(caption), tokenize=False, add_generation_prompt=True)
    assert text.endswith("<|start|>assistant"), text[-40:]
    return text[: -len("<|start|>assistant")] + ANSWER_PREFIX


class Completions:
    """Minimal /v1/completions client (OpenAI-compatible)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, session: str = "samgo"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=8)
        self.model, self.session = model, session

    def line(self, prompt: str, temperature: float, top_k: int, max_tokens: int) -> tuple[str, str]:
        r = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            stop=["\n", END], extra_body={"top_k": top_k, "prompt_cache_key": self.session})
        c = r.choices[0]
        return c.text, c.finish_reason

    def candidates(self, prompt: str, temperature: float, top_k: int, max_tokens: int, k: int) -> list:
        """k sampled next lines; None for a sample that ended the answer."""
        r = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, n=k,
            stop=["\n", END], extra_body={"top_k": top_k, "prompt_cache_key": self.session})
        return [c.text if c.text.strip() else None for c in r.choices]

    def lines(self, prompt: str, temperature: float, top_k: int, max_tokens: int) -> tuple[list[str], bool]:
        """Complete lines (a cut-off last line is dropped) and whether the answer ended after them."""
        r = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            stop=[END], extra_body={"top_k": top_k, "prompt_cache_key": self.session})
        c = r.choices[0]
        return split_lines(c.text, ended=c.finish_reason == "stop")


def split_lines(text: str, ended: bool) -> tuple[list[str], bool]:
    """Whole lines of a sampled chunk; the last one counts only if a newline or the end of answer closed it."""
    parts = text.split("\n")
    complete = parts[:-1] + ([parts[-1]] if ended and parts[-1].strip() else [])
    return [p for p in complete if p.strip()], ended


def generate(llm, prompt: str, cfg: Config = Config(), on_event=None, structure=None, parse_line=None) -> Result:
    """structure/parse_line default to BrickGPT's 8-brick rules; pass brickgpt.v2.Structure2() and
    samgo.palette.parse for the 35-part palette."""
    started = time.time()
    s = structure if structure is not None else Structure()
    parse_line = parse_line or parse
    res = Result(s, False)

    def emit(kind: str, **kw):
        event = {"t": round(time.time() - started, 2), "kind": kind, **kw}
        res.events.append(event)
        if on_event:
            on_event(event)

    refusals = backtracks = 0
    stuck_at, cut = None, cfg.backtrack_parts
    while True:
        done = False
        while len(s.bricks) < cfg.max_bricks and not done:
            context = prompt + "".join(line + "\n" for line in s.txt().splitlines())
            seen_bad: set[str] = set()  # lines already rejected at this position
            repeats = 0
            for attempt in range(cfg.max_brick_rejections):
                if repeats > cfg.max_repeats:
                    if backtracks < cfg.max_backtracks and s.bricks:
                        backtracks += 1
                        cut = cut * 2 if stuck_at is not None and len(s.bricks) <= stuck_at else cfg.backtrack_parts
                        stuck_at = len(s.bricks)
                        keep = max(len(s.bricks) - cut, 0)
                        emit("backtrack", removed=len(s.bricks) - keep, to=keep, n=backtracks)
                        s.truncate(keep)
                        break  # recompute the context from the shorter build
                    res.gave_up = "stuck repeating rejected parts"
                    emit("give_up", why=res.gave_up, at=len(s.bricks))
                    done = True
                    break
                temp = min(cfg.temperature + cfg.temperature_increase * attempt + cfg.repeat_temperature_jump * repeats,
                           cfg.max_temperature)
                if cfg.best_of > 1 and callable(getattr(llm, "candidates", None)) and hasattr(s, "score"):
                    cands = llm.candidates(context, temp, cfg.top_k, cfg.max_line_tokens, cfg.best_of)
                    fresh = [c for c in cands if c is None or c.strip() not in seen_bad]
                    if not fresh:  # every candidate was already rejected here
                        repeats += 1
                        res.calls += 1
                        emit("repeat", lines=[c.strip()[:40] for c in cands if c], temperature=round(temp, 2))
                        continue
                    cands = fresh
                    ends = [c for c in cands if c is None]
                    scored = sorted(((s.score(c), c) for c in cands if c is not None), reverse=True)
                    if scored and scored[0][0] >= 0 and (scored[0][0] > 0 or not ends):
                        batch, ended = [scored[0][1]], False
                    elif ends:
                        batch, ended = [], True
                    else:
                        batch, ended = [scored[0][1]] if scored else [], False
                elif cfg.lines_per_call > 1 and callable(getattr(llm, "lines", None)):
                    batch, ended = llm.lines(context, temp, cfg.top_k, cfg.lines_per_call * cfg.max_line_tokens)
                else:
                    text, finish = llm.line(context, temp, cfg.top_k, cfg.max_line_tokens)
                    batch, ended = ([text] if text.strip() else []), (not text.strip() and finish == "stop")
                res.calls += 1
                accepted, bad = 0, None
                for text in batch:
                    if len(s.bricks) >= cfg.max_bricks:
                        break
                    why = s.reject_reason(text)
                    if why is not None:
                        bad = (text, why)
                        break
                    s.add(parse_line(text))
                    accepted += 1
                    emit("place", brick=text.strip(), n=len(s.bricks))
                if bad:
                    res.rejections += 1
                    if bad[0].strip() in seen_bad:
                        repeats += 1
                    seen_bad.add(bad[0].strip())
                    emit("reject", line=bad[0].strip()[:40], why=bad[1], temperature=round(temp, 2))
                    if accepted:  # progress was made: the next brick starts its own attempt count
                        break
                    continue
                if ended:
                    if hasattr(s, "allow_end") and not s.allow_end() and refusals < cfg.max_end_refusals:
                        refusals += 1  # too little of the target is built: keep going, like a rejected brick
                        res.rejections += 1
                        emit("reject", line="<end>", why="ended_early", temperature=round(temp, 2))
                        if accepted:
                            break
                        continue
                    done = True
                break
            else:
                res.gave_up = "too many rejections for one brick"
                emit("give_up", why=res.gave_up)
                done = True
        cut = s.first_unstable()
        if cut is None:
            res.stable = bool(s.bricks)
            break
        if res.rollbacks >= cfg.max_regenerations:
            res.gave_up = res.gave_up or "rollback limit"
            emit("give_up", why=res.gave_up)
            break
        res.rollbacks += 1
        emit("rollback", to=cut, removed=len(s.bricks) - cut)
        s.truncate(cut)
    res.seconds = time.time() - started
    emit("done", bricks=len(s.bricks), stable=res.stable, rollbacks=res.rollbacks, rejections=res.rejections)
    return res


def main() -> None:
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("caption")
    ap.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--out", type=Path, default=Path("runs/brickgpt/demo"))
    args = ap.parse_args()
    load_dotenv()
    tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
    llm = Completions(args.base_url, args.model, os.environ.get(args.key_env))
    res = generate(llm, render_prompt(tok, args.caption), on_event=lambda e: print(json.dumps(e), flush=True))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bricks.txt").write_text(res.structure.txt() + "\n")
    (args.out / "model.ldr").write_text(res.structure.ldr(args.caption[:40]))
    (args.out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in res.events) + "\n")
    print(f"{len(res.structure.bricks)} bricks, stable {res.stable}, {res.rollbacks} rollbacks, "
          f"{res.rejections} rejections, {res.calls} calls, {res.seconds:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
