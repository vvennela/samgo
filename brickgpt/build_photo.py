"""Photo -> *your* object in bricks: caption (Spark Open) + shape (SAM 3.1 + SAM 3D voxels, from S1/S2) ->
Glimmer places every part, shape-checked (brickgpt/shape.py) -> colored LDraw file + fidelity scores.

    python -m brickgpt.build_photo runs/s6/chair --palette brickgpt --backend sampler --model accounts/vvennela/models/brickgpt-v1-tuned
    python -m brickgpt.build_photo runs/s6/chair --backend completions --model <deployment model>

The folder needs photo.png and target.npz (samgo.pipeline writes both). Output goes to runs/build/<name>/.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    from .caption import caption
    from .decode import ANSWER_PREFIX, Completions, Config, generate
    from .shape import ShapeChecked, fit_target

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--palette", choices=("brickgpt", "v2"), default="brickgpt")
    ap.add_argument("--backend", choices=("completions", "sampler", "replay"), default="completions",
                    help="replay: feed lines from --replay (offline plumbing test, no model)")
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--best-of", type=int, default=3, help="candidates per part, best fills the most of the object")
    ap.add_argument("--tolerance", type=int, default=1, help="cells a part may extend past the object's edge")
    ap.add_argument("--min-coverage", type=float, default=0.7, help="share of the object built before stopping")
    ap.add_argument("--caption", default=None, help="skip Spark and use this caption")
    ap.add_argument("--budget", type=float, default=None, help="sampler: stop calling the API at this worst-case USD estimate")
    args = ap.parse_args()
    load_dotenv(REPO / ".env")

    out = REPO / "runs/build" / args.folder.name
    out.mkdir(parents=True, exist_ok=True)
    text = args.caption or asyncio.run(caption(args.folder / "photo.png"))
    (out / "caption.txt").write_text(text + "\n")
    raw = np.load(args.folder / "target.npz")["target"]

    if args.palette == "v2":
        from samgo.palette import parse

        from .v2 import LAYERS, WORLD, Structure2, messages
        target, base = fit_target(raw, (WORLD, WORLD, LAYERS), 1), Structure2()
    else:
        from .prompts import messages
        from .structure import WORLD, Structure, parse
        target, base = fit_target(raw, (WORLD, WORLD, WORLD), 3), Structure()
    s = ShapeChecked(base, target, parse, tolerance=args.tolerance, min_coverage=args.min_coverage)

    tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
    prompt = tok.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
    prompt = prompt[: -len("<|start|>assistant")] + ANSWER_PREFIX
    if args.backend == "replay":
        llm = Replay(args.replay.read_text().splitlines())
    elif args.backend == "sampler":
        from .fw_sampler import Sampler
        llm = Sampler(args.model, budget_usd=args.budget)
    else:
        if not args.model:
            raise SystemExit("--model is required with --backend completions")
        llm = Completions(args.base_url, args.model, os.environ.get("FIREWORKS_API_KEY"), session=args.folder.name)

    events = open(out / "events.jsonl", "w")
    res = generate(llm, prompt, Config(best_of=args.best_of), structure=s, parse_line=parse,
                   on_event=lambda e: (events.write(json.dumps(e) + "\n"), events.flush()))
    events.close()
    (out / "bricks.txt").write_text(s.txt() + "\n")
    (out / "model.ldr").write_text(s.ldr(args.folder.name))
    summary = {"caption": text, "palette": args.palette, "model": args.model or "base", "best_of": args.best_of,
               "parts": len(s.bricks), "stable": res.stable, "rejections": res.rejections, "rollbacks": res.rollbacks,
               "calls": res.calls, "seconds": round(res.seconds, 1), "gave_up": res.gave_up, **s.metrics()}
    if hasattr(llm, "cost"):
        summary |= {"est_cost_usd_max": round(llm.cost(), 3), "prefill_tokens": llm.prefill_tokens,
                    "sample_tokens": llm.sample_tokens, "budget_hit": llm.over_budget()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if s.bricks:
        render_build(s, out / "model.png", f"{args.folder.name}: IoU {summary['iou']}, {len(s.bricks)} parts")
    print(json.dumps(summary, indent=2))


class Replay:
    """Test-only stand-in for the model: answers each request with the next line of a file ('' ends)."""

    def __init__(self, lines: list[str]):
        self.lines, self.i = lines, 0

    def line(self, prompt, temperature, top_k, max_tokens):
        text = self.lines[self.i] if self.i < len(self.lines) else ""
        self.i += 1
        return text, "stop"

    def candidates(self, prompt, temperature, top_k, max_tokens, k):
        text, _ = self.line(prompt, temperature, top_k, max_tokens)
        return [text or None]


def render_build(s, path: Path, title: str) -> None:
    """Picture of the build in the photo's colors (v1 bricks drawn as v2 bricks: a layer is 3 plates)."""
    from samgo.palette import Part
    from samgo.parts import COLORS

    from .render import draw

    rgb = {code: "#%02x%02x%02x" % c for code, c in ((v[0], v[1]) for v in COLORS.values())}
    parts = [p if hasattr(p, "box") else Part("brick", p.h, p.w, p.x, p.y, 3 * p.z) for p in s.bricks]
    draw(parts, str(path), title, colors=[rgb.get(code, "#9aa3ad") for code in s.colors()])


if __name__ == "__main__":
    main()
