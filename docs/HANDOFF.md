# SAMgo handoff (Nov 4 2026)

Read this first in a fresh session: what SAMgo is, the rules, how to run things, the models and the benchmarks.
Submission: **Wed Nov 4 2026, 6 PM PT** (Meta hackathon). The video plan is in `video_handoff.md`.

## 1. What SAMgo is

Photo -> verified LEGO model with working mechanisms.

- **SAM 3.1 + SAM 3D** (fal) turn one photo into a mesh, with the glass Spark names marked; `samgo/voxelize.py`
  makes a voxel target and code lays it out as real plates and bricks.
- **Muse Spark** (Baseten, `meta-models/muse-spark-open-1026`) knows: what the object is, its working parts (hinged
  panels, spinning parts, legs, wheels) and where each goes, mirror twins, glass, colour; after the build it reviews
  the model against the real object. `samgo/spark_mechanisms.py`, `samgo/part_layout.py`, `samgo/spark_review.py`,
  `samgo/spark_gear.py`, `samgo/spark_glass.py`, `samgo/car_layout.py`.
- **Muse Glimmer 30B**, LoRA fine-tuned on Fireworks, does: it gets the parts near a spot and one generic sentence and
  writes the edit list (`clear` / `remove` / `set` / `add`).
- **Code verifies everything**: real parts, no overlaps, every part held by studs (joint halves included), axles only
  through holes, mechanisms sweep clear, force balance (stud clutch 250), certified step-by-step build
  (`samgo/manual.py`, exact rational witnesses), LDraw export, catalogue colours. The checker's findings go back to
  Glimmer as exact sentences (`samgo/free_edit_test.explain`), up to 3 rounds.

## 2. Standing rules from the user

- **Spark knows, Glimmer does, code verifies.** Anything visible in the photo (wheels, doors, front/back, glass,
  working parts) comes from Spark.
- **No hard-coding to a test object.** The 747, Cessna, helicopter, fan, turbine are held-out exams: never trained or
  tuned on. A gap becomes a generic checker rule plus generic training data.
- Design before code on big changes; propose, get a yes, then build.
- Commit after each part; **do not push**; **never merge into main**; no Claude attribution anywhere.
- Keys stay in `.env`; never print them. Fireworks mutations via the SDK only.
- **Fireworks has one Glimmer slot: always delete training sessions** (section 6).
- Track background jobs with run_in_background or Monitor (no bare `&`). Share builds in 3D via `samgo/viewer3d.py`.
- AWS only if asked. No license/citation lectures. RLForge is a separate earlier project.

## 3. Environment and commands

- Repo: `/Users/vishnuv/Documents/Projects/samgo`, branch **`r4-generalize`**.
- Tests: `UV_PROJECT_ENVIRONMENT=~/.venvs/samgo uv run pytest -q`.
- Anything that samples Glimmer uses the Fireworks SDK venv: `PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m ...`
  (evals: `PYTHONPATH=<repo>:<fireworks cookbook>`, see `runs/r13_tests.sh`).
- Training: `brickgpt/train_serverless.py`, run from the cookbook dir by `runs/rN_chain.sh`; it resumes from
  `runs/train/<run-id>/state.json`.
- Any object: `python -m samgo.pipeline photos/<x>.jpg --prompt <x> --static --assembly learned --model <sampler>
  -o runs/<dir>` (`--size large` or a stud count for bigger models).
- Cars: `python -m samgo.pipeline photos/<car>.png --prompt car --car --assembly learned --model <sampler> --doors
  --buildable [--manual] [--size large] -o runs/<dir>/<car>`.
- 3D viewer: `python -m samgo.viewer3d out.html "Label=path/build.json" ...`. Build kit:
  `python -m samgo.build_kit <artifacts> <out>`.

## 4. Models

| Run | Sampler (Fireworks) | Adds |
|---|---|---|
| r4 | `vvennela/run-3350162e7eff44a7b15fda2dd54c4d05/objects-r4c-s8192-9daf3b92` | object families, place/repair |
| r5 | `vvennela/run-d4f382b536e54966b13e66e9e3274bcf/objects-r5-tuned-5421f012` | doors, cars |
| r6 | `vvennela/run-f6e6e9a6548043ce90afaf9e3b027ad6/objects-r6-tuned-a17f37b4` | hinge, leg |
| r7 | `vvennela/run-cc61285fad3a4326bc96e7d68c6ee1e8/objects-r7-tuned-8d1ffcd0` | hinge with `clear` |
| r9 | `vvennela/run-278c4e5b3c3a47ff9a395babda7f1c8d/objects-r9-tuned-89f8cfcb` | realistic sheets, own-answer feedback rows |
| r10 | `vvennela/run-261ba12dc52b48fba4252347dae2bd99/objects-r10-s24576-591aac60` | side rotor |
| **r11 (serving)** | `vvennela/run-736d451107854752b6ba9b0cb0ad5b78/objects-r11-tuned-d5f4be8e` | upright rotor (turntable) |
| r13 (training) | run id `objects-r13s` | all four mechanism skills on 1.5-2.5x larger bodies |

All: LoRA rank 32, batch 64, 2 epochs, max-seq-len 24576, base Glimmer. Per-run notes:
`fireworks-training-runs/objects-rN/run.md`.

## 5. Benchmarks

Frozen test sets in `runs/audit/`, never regenerated (`gen_*` refuse to overwrite). First try / within 3 rounds.

| Test | r8 | r9 | r10 | **r11** |
|---|---|---|---|---|
| Side rotor `r10-rotors` (30) | 1 | - | 30 / 30 | 28 |
| Upright rotor `r11-vrotors` (30) | - | - | 8 | **19 / 24** |
| Flap sheets `r9-sheets` (30) | 14 / 15 | 15 / 22 | 15 / 19 | 14 / 20 |
| Hinge a strip `r6-primitives` (20) | 14 / 15 | 13 / 13 | 14 / 15 | 14 / 15 |
| Leg `r6-primitives` (20) | 19 | 20 | 20 | 20 |
| Cars, running gear (50) | 48 / 49 | 48 / 50 | 45 / 47 | **48 / 50** |
| Doors on unseen vehicles `r5-frozen-doors` (10) | 10 | 9 | 10 | **10** |
| Objects `r4-frozen-v2` (112) | 92 | 88 | 87 | **92** |

Large-body tests for run 13 (`runs/audit/r13-*-large`, 30 each), run 11 baseline (first / within): flaps 15/20,
side rotors 27/27, upright rotors 23/25, any rotor on the blade test 22/23.

What moved the numbers:
1. **The `clear` edit** (`samgo/edits.py`): `{"op":"clear","box":[x0,x1,y0,y1,z0,z1]}` empties a box, cuts crossing
   parts and re-lays the rest as real plates. Glimmer makes the decision, code does the cell bookkeeping: hinges on
   unseen bodies 4/20 -> 15/20 with ~300-token answers.
2. **Precise feedback** names the part to remove, what a swing hits, which joint half grips nothing; training rows
   built from Glimmer's own answers -> feedback -> fix teach it to use the sentences.
3. **Physical checks**: `kinematics.dangling` (a joint half whose studs grip nothing fails); data regenerated under
   the stricter check.
4. **Generic rotor skills**: side rotors 1 -> 30 of 30 (run 10), upright rotors 8 -> 24 of 30 (run 11).
5. **Several answers per round**, the checker keeps one that passes.

## 6. Fireworks operations

The chains have a watchdog (no log progress for 900 s -> TERM, then KILL). After a KILL, delete the session:

```python
import os; from pathlib import Path; from dotenv import load_dotenv
load_dotenv(Path("/Users/vishnuv/Documents/Projects/samgo/.env"))
from fireworks.training.sdk import FireworksClient
FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"]).delete_training_session(
    "accounts/vvennela/trainingSessions/<ts-id from the log's 'session'/'resumed' event>")
```

(`runs/fw_delete_sessions.py <log>` does this for every session in a log.) Then rerun the same chain; it resumes from
the last promoted checkpoint (every ~2,000 rows).

## 7. Showcase and demos

- 3D showcase (Camry, supercar, 747, helicopter, Cessna): https://claude.ai/artifact/WDqu8phpQUANxebg8BWncL; models in
  `runs/hand_finished/*_current.json` (the pipeline's builds, mechanisms completed with SAMgo's builders, every one
  verified by the same checker).
- Fan from a photo, end to end: `runs/demo_r10/fan2_v5` (Glimmer's rotor first try in 5 of 5 runs on two photos).
- 40-stud builds: `runs/demo_r11_xl` (wind turbine passes every check and stands; every rotor built first try).
- Camry build kit: `runs/build_kit/camry_red_current` (320 parts, 56 certified steps, 60-page manual).
- Video clips and pipeline stills: `runs/video/`.
- Photo angle advice: one three-quarter view is best for single-image 3D.

## 8. Next

1. Run 13: test on the large-body sets and regressions (`runs/r13_tests.sh`, automatic after training), then the
   40-stud showcase again.
2. Standing for the heaviest large models (more hinges/legs as load grows), flaps and doors at large size.
3. More through Spark: door style (normal / scissor), wings and spoilers.
4. Bigger pieces in the layout (fewer 1x1 plates).
5. Later: convertibles; known-good LEGO sub-assemblies (plan first).
