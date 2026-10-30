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

from .running_gear_task import (FEEDBACK, apply, axles_of, check, edits_from, explain, pose_frame, pose_open, prompt,
                                step_problem)

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
    ap.add_argument("--sequential", action="store_true",
                    help="one axle per step, each on the build as it stands after the previous step")
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
    for folder in args.folders:
        folder = REPO / folder if not folder.is_absolute() else folder
        from .arches import build_arched_car
        from .voxelize import load_mesh, voxelize
        car = build_arched_car(voxelize(load_mesh(folder / "object.glb"), 20), attempts=6)
        parts, target = car.parts, car.target
        problem = pose_frame(parts, target, axles_of(parts))  # frame-and-shell car: the gear is only added
        out = REPO / "runs/demo" / folder.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "input.json").write_text(json.dumps(problem["full"]))
        (out / "reference.json").write_text(json.dumps(parts))
        label = args.label + ("_seq" if args.sequential else "")
        log = {"car": folder.name, "model": label, "steps": []}

        def solve(prob, partial, tag):
            """Ask, check, feed errors back; returns the build after the best answer and whether it passed."""
            chat = prompt(prob)
            result = None
            seen = set()
            for rnd in range(1, args.rounds + 1):
                for temp in (0.0, 0.7, 0.7):  # an answer it already gave is resampled with some randomness
                    p = sdk.SamplingParams(max_tokens=args.max_tokens, temperature=temp,
                                           stop=renderer.get_stop_sequences())
                    toks = list(client.sample(prompt=renderer.build_generation_prompt(chat), num_samples=1,
                                              sampling_params=p).result(timeout=900).sequences[0].tokens)
                    msg, _ = renderer.parse_response(toks)
                    text = get_text_content(msg)
                    if text not in seen:
                        break
                seen.add(text)
                chat.append({"role": "assistant", "content": text})
                try:
                    edits = edits_from(text)
                    ok, issues = explain(prob, edits, partial=partial)
                    result = apply(prob, edits)
                except Exception as e:  # noqa: BLE001
                    ok, issues, edits = False, [f"could not use the answer: {e}"], []
                log["steps"].append({"step": tag, "round": rnd, "passed": ok, "edits": len(edits), "tokens": len(toks),
                                     "issues": issues[:4]})
                print(json.dumps({"car": folder.name, "model": label, "step": tag, "round": rnd, "passed": ok,
                                  "edits": len(edits), "issues": issues[:2]})[:400], flush=True)
                if ok:
                    return result, True
                chat.append({"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))})
            return result, False

        if args.sequential:
            sizes = {p["x"]: p["size"] for p in parts if p.get("type") == "wheel"}
            build = problem["full"]
            edges = axles_of(parts)
            for k, edge in enumerate(edges):
                step = step_problem(build, target, edge, sizes[edge], frame=True)
                new, ok = solve(step, partial=k < len(edges) - 1, tag=f"axle at x={edge}")
                if not ok or new is None:
                    break
                build = new                                  # the next step sees what this one built
            final_ok = check(build, target)[0] if build is not problem["full"] else False
            (out / f"{label}.json").write_text(json.dumps(build))
        else:
            build, final_ok = solve(problem, partial=False, tag="both axles")
            if build is not None:
                (out / f"{label}.json").write_text(json.dumps(build))
        log["passed"] = bool(final_ok)
        print(json.dumps({"car": folder.name, "model": label, "final_passed": bool(final_ok)}), flush=True)
        (out / f"{label}_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
