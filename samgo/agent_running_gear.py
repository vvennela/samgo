"""Can untrained Glimmer make a car roll? A running-gear test with verifier feedback (mechanisms.md, layer 1).

Each test car is a verified algorithmic build (S4f) with its axles and wheels removed and its Technic chassis
bricks turned into plain bricks, so a correct answer is known to exist. Glimmer gets the parts, the car's size,
a parts-and-rules sheet and the task; it answers with edits. The edited build goes through the structural (S3a),
function (S3c) and stability (S3b) verifiers, and their error text goes back to Glimmer, up to --rounds times.

    uv run python -m samgo.agent_running_gear runs/s6/car runs/s6/beetle --rounds 3
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .arches import build_arched_car
from .stability import stability
from .technic import structural, verify_function
from .verifier import MatchConfig, verify

MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"

RULES = """You are a LEGO Technic engineer. You edit a LEGO model, given as a numbered list of parts (JSON).

Coordinates: x and y in studs (1 stud = 8 mm), z in plates (1 plate = 3.2 mm; a brick is 3 plates tall).
A part {"w": W, "l": L, ...} fills studs x..x+W-1 along x, y..y+L-1 along y, from height z.
Part types:
- {"type": "plate"|"brick", "w", "l", "x", "y", "z", "color"}: plain parts. Parts connect only by studs: one part
  sitting directly on another, sharing at least one stud.
- {"type": "technic_brick", "w": N, "l": 1, "x", "y", "z", "color", "hole": "round"|"axle"}: a 1xN brick (N in 2, 4,
  6, 8) with N-1 holes. The holes go ACROSS the brick (along y for a brick running along x), at the stud edges
  between its studs (x+1 ... x+N-1), 5.8 mm above its bottom. A "round" hole lets an axle turn; an "axle" hole
  locks it.
- {"type": "axle", "axis": "y", "x": E, "z": Z, "y": Y0, "length": L}: a rod along y on stud edge x=E, at the hole
  height of Technic bricks whose bottom is at plate z=Z, from y=Y0 to y=Y0+L. It must pass through at least 2
  aligned round holes, and through no other solid part.
- {"type": "wheel", "axis": "y", "x": E, "z": Z, "y": Y0, "width": 2}: a 30.4 mm wheel with tyre on that axle,
  occupying y=Y0..Y0+2. It must sit on an axle, must not hit any part when it turns, and the wheels must be the
  lowest thing touching the ground.
A car rolls when it has at least 2 parallel axles, each with a wheel on each side, turning in round holes."""

TASK = """This model is a car body. Its length runs along x (x = {x0}..{x1}), its width along y (y = {y0}..{y1}).
It has no running gear yet. Make it a car that rolls: add axles and wheels, and change parts where needed (for
example, turn plain chassis bricks into Technic bricks with round holes where an axle must pass). Keep the body
as it is otherwise.

Answer with a JSON list of edits, and nothing after it:
[{{"op": "set", "index": I, "part": {{...}}}}, {{"op": "add", "part": {{...}}}}, {{"op": "remove", "index": I}}]

Parts:
{parts}"""

FEEDBACK = """The verifiers checked your edited model:
{issues}
Fix these problems. Answer with the complete JSON list of edits again (it replaces your previous answer; edits
still refer to the original numbered parts)."""


def problem(folder: Path):
    """(body parts given to the model, structural target, the reference build): the arched car with its running
    gear taken off and its Technic beams turned back into plain bricks."""
    car = build_arched_car(np.load(folder / "target.npz")["target"], attempts=6)
    body = [p for p in car.parts if p.get("type") not in ("axle", "wheel", "bush")]
    body = [({k: v for k, v in p.items() if k != "hole"} | {"type": "brick"}) if p["type"] == "technic_brick" else p
            for p in body]
    return body, car.target, car.parts


def apply(body: list[dict], edits: list[dict]) -> list[dict]:
    parts: list[dict | None] = list(body)
    added = []
    for e in edits:
        if e.get("op") == "set":
            parts[int(e["index"])] = e["part"]
        elif e.get("op") == "remove":
            parts[int(e["index"])] = None
        elif e.get("op") == "add":
            added.append(e["part"])
    return [p for p in parts if p is not None] + added


def check(parts: list[dict], target: np.ndarray) -> tuple[bool, list[str], dict]:
    issues, result = [], {}
    try:
        s = verify(structural(parts), target, config=MatchConfig())
        f = verify_function(parts)
    except Exception as e:  # noqa: BLE001  a malformed part
        return False, [f"invalid model: {e}"], {"error": str(e)}
    issues += [f"structure: {i.message}" for i in s.issues] + [f"function: {i.message}" for i in f.issues]
    result |= {"structure": s.passed, "function": f.passed, "iou": round(s.metrics.get("iou", 0.0), 3)}
    if s.passed and f.passed:
        # Reported, not required: some bodies already fail stability on their own (the truck's raised bed),
        # which is not a running-gear problem.
        result["stands"] = stability(parts, vehicle=True).passed
    return not issues, issues, result


_CHAT = None
SAMPLER_MODEL = None  # a sampler checkpoint path (<account>/<run_id>/<name>) to test a fine-tuned model; None = base


def ask(messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """Base Glimmer through the Fireworks training SDK's sampler (it is not on the serverless chat endpoint),
    with the model's own chat template. Run from the cookbook root with this repo on PYTHONPATH."""
    global _CHAT
    from fireworks.training import sdk

    if _CHAT is None:
        from fireworks.training.sdk import FiretitanServiceClient
        from training.utils.serverless import serverless_base_url
        from training.utils.supervised import build_renderer
        from training.utils.tokenizers import load_tokenizer

        tok = load_tokenizer(HF_MODEL)
        service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                         base_url=serverless_base_url("https://api.fireworks.ai"))
        _CHAT = (tok, build_renderer(tok, HF_MODEL, "muse_glimmer"),
                 service.create_sampling_client(tokenizer=tok, **({"model_path": SAMPLER_MODEL} if SAMPLER_MODEL
                                                                  else {"base_model": MODEL})))
    tok, renderer, client = _CHAT
    from training.renderer import get_text_content

    prompt = renderer.build_generation_prompt(messages)
    params = sdk.SamplingParams(max_tokens=max_tokens, temperature=0.6, stop=renderer.get_stop_sequences())
    out = client.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=600)
    tokens = list(out.sequences[0].tokens)
    msg, _ = renderer.parse_response(tokens)
    return get_text_content(msg), {"prompt_tokens": len(prompt.to_ints()), "completion_tokens": len(tokens)}


def edits_from(text: str) -> list[dict]:
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    if not m:
        raise ValueError("no JSON list of edits in the answer")
    return json.loads(m.group(0))


def run(folder: Path, rounds: int, max_tokens: int, out: Path) -> dict:
    body, target, reference = problem(folder)
    solid = [p for p in body if p.get("type") in ("plate", "brick")]
    xs = [v for p in solid for v in (p["x"], p["x"] + p["w"] - 1)]
    ys = [v for p in solid for v in (p["y"], p["y"] + p["l"] - 1)]
    ok_ref, ref_issues, _ = check(reference, target)
    numbered = "\n".join(f"{i}: {json.dumps(p)}" for i, p in enumerate(body))
    messages = [{"role": "system", "content": RULES},
                {"role": "user", "content": TASK.format(x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys), parts=numbered)}]
    log = {"car": folder.name, "parts": len(body), "reference_passes": ok_ref, "rounds": []}
    for k in range(1, rounds + 1):
        t = time.time()
        try:
            text, usage = ask(messages, max_tokens)
        except Exception as e:  # noqa: BLE001
            log["rounds"].append({"round": k, "error": f"api: {e}"[:300]})
            break
        rec = {"round": k, "seconds": round(time.time() - t, 1), "usage": usage}
        try:
            edits = edits_from(text)
            parts = apply(body, edits)
            passed, issues, result = check(parts, target)
            rec |= {"edits": len(edits), "passed": passed, **result, "issues": issues[:12]}
        except Exception as e:  # noqa: BLE001
            passed, issues = False, [f"could not use the answer: {e}"]
            rec |= {"passed": False, "issues": issues, "answer_tail": text[-400:]}
        log["rounds"].append(rec)
        print(json.dumps({"car": folder.name, **{k2: v for k2, v in rec.items() if k2 != "usage"}})[:600], flush=True)
        if passed:
            (out / f"{folder.name}_parts.json").write_text(json.dumps(parts))
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues[:20]))}]
    log["passed"] = bool(log["rounds"]) and log["rounds"][-1].get("passed", False)
    log["rounds_used"] = len(log["rounds"])
    return log


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("-o", "--out", type=Path, default=Path("runs/agent/running_gear"))
    ap.add_argument("--model", default=None, help="sampler checkpoint path of a fine-tuned model (default: base)")
    args = ap.parse_args()
    global SAMPLER_MODEL
    SAMPLER_MODEL = args.model
    args.out.mkdir(parents=True, exist_ok=True)
    logs = [run(f, args.rounds, args.max_tokens, args.out) for f in args.folders]
    (args.out / "results.json").write_text(json.dumps(logs, indent=2))
    for lg in logs:
        print(f"{lg['car']:8s} reference passes: {lg['reference_passes']}  Glimmer passed: {lg['passed']} "
              f"after {lg['rounds_used']} round(s)")


if __name__ == "__main__":
    main()
