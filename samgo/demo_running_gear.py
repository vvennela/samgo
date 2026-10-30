"""Live running-gear demo: a photo car with no running gear, solved by a model with up to N feedback rounds.

For each photo car folder (object.glb from SAM 3D), builds the verified car, poses the open problem (both
axles stripped, nothing marks the beam positions), asks the model, checks with the actionable messages, feeds
errors back, and saves the input body and the model's final build for the 3D viewer.

    python -m samgo.demo_running_gear runs/s6/camry runs/s6/beetle --model <sampler path> --label r2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .running_gear_task import FEEDBACK, apply, axles_of, edits_from, explain, pose_open, prompt

REPO = Path(__file__).resolve().parent.parent
MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="base")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import get_text_content
    from training.utils.serverless import serverless_base_url
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    tok = load_tokenizer(HF_MODEL)
    renderer = build_renderer(tok, HF_MODEL, "muse_glimmer")
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url("https://api.fireworks.ai"))
    client = service.create_sampling_client(tokenizer=tok, **({"model_path": args.model} if args.model
                                                               else {"base_model": MODEL}))
    params = sdk.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    for folder in args.folders:
        folder = REPO / folder if not folder.is_absolute() else folder
        from .arches import build_arched_car
        from .voxelize import load_mesh, voxelize
        car = build_arched_car(voxelize(load_mesh(folder / "object.glb"), 20), attempts=6)
        parts, target = car.parts, car.target
        problem = pose_open(parts, target, axles_of(parts))
        out = REPO / "runs/demo" / folder.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "input.json").write_text(json.dumps(problem["full"]))
        (out / "reference.json").write_text(json.dumps(parts))
        chat = prompt(problem)
        log = {"car": folder.name, "model": args.label, "rounds": []}
        for rnd in range(1, args.rounds + 1):
            toks = list(client.sample(prompt=renderer.build_generation_prompt(chat), num_samples=1,
                                      sampling_params=params).result(timeout=900).sequences[0].tokens)
            msg, _ = renderer.parse_response(toks)
            text = get_text_content(msg)
            chat.append({"role": "assistant", "content": text})
            try:
                edits = edits_from(text)
                ok, issues = explain(problem, edits)
                result = apply(problem, edits)
            except Exception as e:  # noqa: BLE001
                ok, issues, edits, result = False, [f"could not use the answer: {e}"], [], None
            log["rounds"].append({"round": rnd, "passed": ok, "edits": len(edits), "tokens": len(toks),
                                  "issues": issues[:4]})
            print(json.dumps({"car": folder.name, "model": args.label, "round": rnd, "passed": ok,
                              "edits": len(edits), "issues": issues[:2]})[:400], flush=True)
            if result is not None:
                (out / f"{args.label}.json").write_text(json.dumps(result))
            if ok:
                break
            chat.append({"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))})
        (out / f"{args.label}_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
