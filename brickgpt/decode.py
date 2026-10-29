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


def generate(llm, prompt: str, cfg: Config = Config(), on_event=None) -> Result:
    started = time.time()
    s = Structure()
    res = Result(s, False)

    def emit(kind: str, **kw):
        event = {"t": round(time.time() - started, 2), "kind": kind, **kw}
        res.events.append(event)
        if on_event:
            on_event(event)

    while True:
        done = False
        while len(s.bricks) < cfg.max_bricks and not done:
            context = prompt + "".join(b.txt() + "\n" for b in s.bricks)
            for attempt in range(cfg.max_brick_rejections):
                temp = min(cfg.temperature + cfg.temperature_increase * attempt, cfg.max_temperature)
                text, finish = llm.line(context, temp, cfg.top_k, cfg.max_line_tokens)
                res.calls += 1
                if not text.strip() and finish == "stop":  # the model ended its answer
                    done = True
                    break
                why = s.reject_reason(text)
                if why is None:
                    b = parse(text)
                    s.add(b)
                    emit("place", brick=b.txt(), n=len(s.bricks))
                    break
                res.rejections += 1
                emit("reject", line=text.strip()[:40], why=why, temperature=round(temp, 2))
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
