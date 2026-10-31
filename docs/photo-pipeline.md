# Photo pipeline with learned assembly

The pipeline builds a procedural car body and derives tire regions and sizes, then asks Glimmer to add the running gear. Model output goes through the same edit executor and checker used by evaluation and feedback. Only accepted proposals advance the build. There is no procedural replacement for a failed model answer.

## Commands

Install the declared local dependencies in the external environment. `--inexact` preserves separately installed training packages:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/samgo"
uv sync --locked --inexact

# Procedural reference mode; fresh perception needs MODEL_API_KEY and FAL_KEY.
uv run --no-sync python -m samgo.pipeline photos/beetle.jpg --prompt car --car \
  --assembly procedural -o runs/photo-beetle
```

Live learned mode needs the existing Fireworks training SDK, its `training` cookbook module, Glimmer tokenizer access, and `FIREWORKS_API_KEY` in the environment or repository `.env`. These are the same dependencies as the earlier training/demo scripts; they are not installed by the base project. On the development machine, this is the tested runtime:

```bash
export SAMGO_COOKBOOK="$HOME/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0"
export SAMGO_TRAINING_PYTHON="$HOME/.venvs/fwcookbook/bin/python"

PYTHONPATH=".:$SAMGO_COOKBOOK" uv run --no-project \
  --python "$SAMGO_TRAINING_PYTHON" --with fal-client --with meta-sam-parser \
  python -m samgo.pipeline photos/beetle.jpg --prompt car --car --assembly learned \
  --mesh runs/s6/beetle/object.glb \
  --model vvennela/run-6a81b005f4ba4a039e8dda1760edf3b9/running-gear-r3-tuned-ef4d7fa6 \
  --rounds 3 --max-tokens 4000 --timeout 60 -o runs/photo-beetle-learned
```

The specific checkpoint and cookbook installation above must remain available. Omit `--model` to use the configured base Glimmer model. `--mesh` explicitly supplies an existing mesh and records its source and hash; the pipeline does not claim to have verified that it belongs to the photo. Omit `--mesh` for perception from the photo. Add `--fresh` to bypass both perception and function-spec caches. `--car` skips function inference; without `--car` or `--static`, the existing Spark function-spec service is used.

By default, assembly proceeds one axle line at a time. The procedural reference must pass the partial-task checker before that task is offered. If it cannot (for example because the missing rear gear exceeds the full-target similarity allowance), the run explicitly selects the already verified joint task and records why. `--joint` requests both axle lines in one task directly. Learned assembly currently supports cars only.

`--rounds` bounds proposals per task, with one sampling request per round; temperature is 0 on the first round and 0.7 afterward. `--timeout` bounds waiting for a sampling response, not tokenizer/runtime initialization. Function inference and perception have 120-second and 300-second stage deadlines. Failed/timeout requests consume a round. These deadlines do not guarantee cancellation of remote work already submitted. Token counts and whether a response reached its token limit are recorded; reaching the limit is not itself proof of truncation.

## Outputs and replay

`OUT/summary.json` always points to the latest run. Each invocation writes into its own `OUT/artifacts/<run-id>/` directory, including failed runs, so a failed rerun cannot publish old instructions as its result.

| Artifact | Contents |
|---|---|
| `summary.json` | Mode, outcome, source/code/dependency hashes or versions, stage timings, artifact paths, cache provenance |
| `function_spec.json` | Explicit vehicle/static choice or inferred spec |
| `object.glb`, `target.npz`, `target.png` | Mesh used and original voxel target |
| `body_search.json` | Up to three procedural seeds, six layout attempts per beam size, with full validation outcomes |
| `assembly_target.npz` | Target after procedural shaping; separate from the original target |
| `assembly/trace.json` | Sampler identity, budgets, task mode, input/local numbering, full model-visible message histories, raw answers, parsed edits, rejected candidates, feedback, accepted states, tokens and timings |
| `assembly/input.json`, `reference.json`, `accepted.json`, `target.npz` | Stripped body, procedural reference, last accepted state and checked target |
| `build.json`, `checks.json` | Exact final accepted state and structure/function/stability reports, plus original-target fidelity |
| `viewer.html` | That exact build, labeled with mode, live/replay source, and check outcome |
| `instructions/` | Inventory and schematic steps, emitted only after assembly and all required checks pass |

The viewer is static in this path: unchecked motion animations are disabled. Instructions still do not establish insertion access, temporary support, or physical buildability. Original-target fidelity is reported separately and does not replace the assembly-target gate.

For a network-free rehearsal, replay an `assembly/trace.json` from an earlier run:

```bash
uv run --no-sync python -m samgo.pipeline photos/beetle.jpg --prompt car --car \
  --assembly learned --mesh runs/s6/beetle/object.glb \
  --replay /absolute/path/to/previous/artifacts/RUN/assembly/trace.json \
  -o runs/photo-beetle-replay
```

Replay requires the same reconstructed reference, target and complete message history. Changed inputs, changed feedback, or exhausted recordings fail explicitly. It is labeled `replay` and is not counted as fresh model inference. Sampling/runtime errors in a recording cannot be replayed as model answers. A changed dependency or checker can invalidate a recording; use the recorded source/dependency metadata to diagnose this rather than relaxing the checks.

Perception caches are indexed by photo bytes, segmentation prompt, model IDs, preparation settings and perception source code. Cached mesh contents are checked against their recorded hash. Function-spec caches use photo/model/code identity. Existing legacy `OUT/object.glb` files are not implicitly trusted; pass them with `--mesh` explicitly. Intermediate geometry and assembly are recomputed each run.

The CLI exits 0 only when required checks and exports succeed; otherwise it exits 1. A failed learned run keeps its last accepted state and diagnostic trace. Failure to construct valid scaffolding is reported before calling the model. The compatibility `samgo.demo_running_gear` command uses the same assembly runner, but retains its legacy output filenames.
