# SAMgo handoff (as of Nov 2 2026, 23:00 PT; sections 0 and 11 are the newest)

Read this first in a fresh session. It is the whole state: what SAMgo is, the rules, how to run things, every
benchmark so far, what was built in the last session, and what is open. Submission: **Wed Nov 4 2026, 6 PM PT**
(Meta hackathon).

## 0. Newest state (Nov 2 night)

- **Latest Glimmer: run 10** (generic rotor skill on top of run 9). Sampler (24,576-row checkpoint, 1.84 of 2 epochs;
  Fireworks lost the session at step 386 and the checkpoints would not warm-start):
  `vvennela/run-261ba12dc52b48fba4252347dae2bd99/objects-r10-s24576-591aac60`.
  Run 9: `vvennela/run-278c4e5b3c3a47ff9a395babda7f1c8d/objects-r9-tuned-89f8cfcb`.
- Results r8 / r9 / r10 (first / within rounds): rotors 1 / - / 30/30; flap sheets `r9-sheets` 14/15, 15/22, 15/19;
  hinge 14/15, 13/13, 14/15; leg 19, 20, 20; cars 48/49, 48/50, 45/47; doors 10, 9, 10; objects 92, 88, 87/111.
  747 sweep r10: flap L3+S3, gear L0-L2.
- New since section 8: Spark names spinning parts (`spark_mechanisms._rotors`, data `samgo/rotor_data.py`), legs
  (`samgo/spark_gear.py`) and glass (`samgo/spark_glass.py`) for any object; `kinematics.dangling` (a joint half that
  grips nothing fails); `kinematics.rotors`; `_near_pairs` broad phase; stability uses interior point first (a
  3,400-part model in seconds instead of hours); `samgo/build_kit.py` (parts.csv, BrickLink wanted list, certified
  steps as pictures); trainer never exports a partial from a failed warm start.
- **Fan (held-out, from a photo) passes end to end** with run 10: `runs/demo_r10/fan2_v5` (Spark: blades spin;
  Glimmer's rotor first try in 5 of 5 runs on two fan photos; structure, connected, stands, rotor turns). Build kits:
  `runs/build_kit/camry` (certified, 49 steps), `runs/build_kit/fan` (schematic; the head overhangs, not certified).
  3D showcase: https://claude.ai/artifact/WDqu8phpQUANxebg8BWncL. Wind turbine: model 3-5 studs deep, no rotor spot.
- Held-out exams: the 92-stud 747 (`runs/s6/boeing747_92`), desk fan and wind turbine photos (`docs/photo-sources.txt`).
  The F-16 photo is local only (war rule: not in the demo or repo).

## 1. What SAMgo is

Photo -> verified LEGO model with working mechanisms.

- **SAM 3.1 + SAM 3D** (fal) turn one photo into a mesh; `samgo/voxelize.py` makes a voxel target.
- **Muse Spark** (Baseten, `meta-models/muse-spark-open-1026`) is the brain: what the object is, its paint colour, and
  for cars its layout (wheel centres, door outline, which end is the front). `samgo/function_spec.py`,
  `samgo/car_layout.py`.
- **Muse Glimmer 30B**, LoRA fine-tuned on Fireworks, is the hands: it edits the parts list to add mechanisms (running
  gear, door hinges, flaps, legs). It answers with a JSON edit list.
- **Code verifies everything**: structure (real parts, no overlaps, connected), force balance (stud clutch 250),
  kinematics (hinges swing clear, wheels spin), a certified build manual (`samgo/manual.py`, exact rational
  witnesses), LDraw export, produced colours from the Rebrickable catalogue.

## 2. Standing rules from the user (do not break)

- **Spark knows, Glimmer does, code verifies.** When code is guessing something visible in the photo (wheel
  positions, doors, front/back, glass, spoilers), route it through Spark instead. The user pushed hard on this.
- **No hard-coding to a test object.** The 747 is the generalization test: never train on it, never tune generators
  against it. Look at failures only to find the missing *generic* skill.
- Design before code on big changes; propose, get a yes, then build. Don't churn proposals.
- Commit after each part; **do not push** unless asked; **never merge into main**; never add a Co-Authored-By line.
- Keys stay in `.env`; never print them. Fireworks mutations via the SDK only.
- **Fireworks has one Glimmer slot: always delete training sessions** (a killed trainer leaves one open; snippet in
  section 7). LoRA SFT on Glimmer is free through Nov 30 Pacific; sampling is billed.
- Track background jobs with run_in_background or Monitor (no bare `&`). Share builds in 3D via `samgo/viewer3d.py`.
- AWS (new: 64 vCPU G/VT Spot quota in us-west-2): only if asked; results to S3 + report, check every 10 min, kill after
  two bad checks, shut down after.
- No license/citation lectures. RLForge is a separate earlier project.

## 3. Environment and commands

- Repo: `/Users/vishnuv/Documents/Projects/samgo`, branch **`r4-generalize`**, 19 commits ahead of `origin/r4-generalize`
  (not pushed). Everything is committed.
- Tests: `UV_PROJECT_ENVIRONMENT=~/.venvs/samgo uv run pytest -q` (323 pass). Don't use the repo `.venv` (iCloud hangs).
- Anything that samples Glimmer needs the Fireworks SDK venv: `PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m ...`
  (the samgo venv lacks `orjson`; evals there fail instantly with "No module named orjson").
- Training script: `brickgpt/train_serverless.py` (run from the cookbook dir, see `runs/r7_chain.sh`). It resumes from
  `runs/train/<run-id>/state.json` automatically.
- Full photo pipeline (car):
  `PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m samgo.pipeline photos/<car>.png --prompt car --car --assembly learned
  --model <sampler> --doors --buildable [--manual] [--size large] -o runs/<dir>/<car>`
  - `--assembly procedural` uses the builder's own verified running gear and hinges (no Glimmer).
  - Spark layout is on by default for cars (`--no-spark-layout` to disable).
- 3D viewer: `PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m samgo.viewer3d out.html "Label=path/build.json" ...` then `open`.
- The Mac can run out of memory with several large builds plus evals at once; Claude Code may kill background jobs.

## 4. Models

| Run | Sampler path (Fireworks) | Trained on |
|---|---|---|
| r4 | `vvennela/run-3350162e7eff44a7b15fda2dd54c4d05/objects-r4c-s8192-9daf3b92` | object families, place/repair |
| r5 | `vvennela/run-d4f382b536e54966b13e66e9e3274bcf/objects-r5-tuned-5421f012` | `data/mix_r5` (5,480 rows: objects, doors, cars) |
| r6 | `vvennela/run-f6e6e9a6548043ce90afaf9e3b027ad6/objects-r6-tuned-a17f37b4` | mix_r5 + `data/primitives_r6` (hinge set-style, leg) |
| **r7 (latest)** | `vvennela/run-cc61285fad3a4326bc96e7d68c6ee1e8/objects-r7-tuned-8d1ffcd0` | mix_r5 + `data/primitives_r7` (3,841: hinge with `clear`, leg) |

All: LoRA rank 32, batch 64, 2 epochs, max-seq-len 24576, base Glimmer.

## 5. Benchmarks

Frozen test sets live in `runs/audit/` and are never regenerated (`gen_*` refuse to overwrite).

| Test | r4 | r5 | r6 | **r7** |
|---|---|---|---|---|
| Objects `r4-frozen-v2`, first try (112) | 87 | 80 | 86 | **87** |
| Doors on unseen vehicles `r5-frozen-doors` (10) | 0 | 7 | 8 | **9** |
| Cars, 50, first / within 3 rounds (`eval_running_gear`) | – | 47 / 49 | 48 / 50 | 48 / 50 |
| Hinge a strip, unseen bodies `r6-primitives` (20) | – | 0 | 4 | **15** |
| Add a leg, unseen bodies `r6-primitives` (20) | – | 3 | 20 | 19 |
| 747 gear, 4 handoff levels | – | 0 | 4 | 3 |
| **747 flap, 4 handoff levels** | – | 0 | 0 | **0** |

Notes:
- Objects by family: small-car gear dropped from 6/6 (r5) to 2/6 (r6) and 1/6 (r7); door place recovered 0 -> 6/6
  from r6; TV repair 2 -> 6/6 from r6. Held-out families (microwave, skateboard) place 0/10 every run.
- r6 and r7 give **identical** answers on all 50 cars (greedy decoding converged; not a cache bug, checked).
- r6's 747 gear was one memorised leg at the same coordinates for every level.
- Results files: `runs/audit/*/results/<run>/`, `runs/audit/r6-primitives/<run>.json`, `runs/eval/r*_cars.json`,
  `runs/granularity/*.json` (sweep per run; `r6-ring.json` = with neighbour ring).

## 6. Key findings (the research story)

1. **Granularity sweep** (`samgo/granularity.py`; L0 whole object -> L1 structure -> L2 small area -> L3 exact line, with
   a ring of neighbouring parts shown at L1-L3): granularity was not the bottleneck. Base Glimmer writes 5-12k-token,
   cut-off answers at every level; fine-tuned Glimmer writes short valid answers at every level. The missing piece was a
   skill: **remove before you add**.
2. **Precise feedback** (`free_edit_test.explain`) names the input part to remove, what a hinge's swing hits, which
   parts are loose. It got r5 to act on feedback (repeats fell from most to 1 of 8), but r7 again repeats the same
   answer on the 747 once it is confident.
3. **Glimmer is bad at cell bookkeeping, good at decisions.** Hand re-tiling around a cut produced overlapping pieces
   (r6 wrote 50-160 `set` edits on the wing). The generic **`clear` edit** (`samgo/edits.py`):
   `{"op":"clear","box":[x0,x1,y0,y1,z0,z1]}` empties an inclusive box in the whole model, cuts crossing parts and
   re-lays the rest as real plates (identity kept on the biggest piece), drops crumbs of 6 cells or fewer left hanging.
   Freeing a panel = clear a channel on every side but the hinge side. This took unseen-body hinges 4/20 -> 15/20 with
   ~300-token answers.
4. **747 flap still 0/4.** r7 places the clip/bar hinge plates *inside* the wing's own plates (overlap) instead of on a
   cleared channel, and repeats the identical answer for all 3 rounds despite feedback.

## 7. Fireworks operations

- Stalls happen (a step taking 15+ min). The chains have a watchdog (no log progress for 900 s -> TERM, then KILL). A
  KILL skips the trainer's cleanup, so **delete the session afterwards**:

```python
import os; from pathlib import Path; from dotenv import load_dotenv
load_dotenv(Path("/Users/vishnuv/Documents/Projects/samgo/.env"))
from fireworks.training.sdk import FireworksClient
FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"]).delete_training_session(
    "accounts/vvennela/trainingSessions/<ts-id from the log's 'session'/'resumed' event>")
```

- Then rerun the same chain; it resumes from the last promoted checkpoint (every 2,048 rows).
- Run 7 needed two resumes (rows 10,240 and 14,336); its final session is deleted. Nothing is running now.

## 8. What was built in the last session (all committed)

- `samgo/free_edit_test.py`: generic `check` and `explain`, `stand=False` skips the balance solve for part tasks;
  `RULES` documents `clear`.
- `samgo/granularity.py`: sweep, `--only`, neighbour ring, wheeled bodies raised (`LIFT = 12`), one leg not asked to stand.
- `samgo/primitives.py`: run 6/7 data generator (hinge-a-strip on generic thin-sheet bodies and car shells, leg under
  a raised body; L1/L2/L3 instructions; realistic mistake -> feedback -> fix rows). `samgo/eval_primitives.py`.
- `samgo/edits.py`: the `clear` op. `tests/test_edits_clear.py`.
- `samgo/glass.py`: windows decided per cell (visible cells of the cabin outline and stepped windshield/rear window
  inside a greenhouse around the roof), mirrored left/right, plates across a window edge cut into glass + body pieces
  (only produced trans-clear sizes; kept only if no part comes loose).
- `samgo/consolidate.py`: merge pass (neighbouring same-colour plates in one rigid body -> bigger real plates; three
  stacked identical plates -> one brick). In the pipeline after paint.
- `samgo/car_layout.py`: Spark reads wheel centres and door outline on a depth-shaded side render with stud ticks
  (cached in `runs/research_cache/layout_*.json`). Pipeline uses it: axles at Spark's wheels when no tyres show (only
  if the wheelbase is 45-80% of the length), door planner tries Spark's outline first and shaves bulges outboard of the
  door face, doors hinge on the front edge (Spark says which end is the front).
- `samgo/arches.py`: axle fallback when no tyres show (GT1). `samgo/doors.py`: outline plan, front-edge hinging via a
  mirrored plan, non-solid mirrored walls rejected. Pipeline: a door cut that can't fit leaves the car without doors
  instead of aborting.

## 9. Car demos

| Car | Where | Status |
|---|---|---|
| Camry, certified (old glass) | `runs/demo_v2/camry` | passes everything, manual certified in 51 steps; glass placement wrong (fixed later) |
| Camry, new glass | `runs/demo_v3/camry` | all model checks pass; **manual does not certify** (stuck on low rear plates; also with merging) |
| Audacious restomod | `runs/demo_large_proc/audacious_rear34` (`photos/audacious_rear34.png`) | **passes everything** at `--size large --assembly procedural --doors --buildable`: 843 parts, front-hinged mirrored doors, glass, silver |
| Porsche 911 GT1 | `runs/demo_large_proc/gt1` (`photos/gt1.png`) | doors swing, function ok; **fails stability**: door too tall (z 1-16) for two swivel hinges, a stud needs 385 vs 250 clutch |
| Audacious side profile | `runs/demo_large_proc/audacious_side` | no door spot fits (side photos give SAM 3D least depth) |

Viewers: `runs/viewer/audacious_front_hinge.html`, `runs/viewer/large_cars.html`, `runs/viewer/camry_v3.html`.
Photo angle advice: one three-quarter view is best; a straight front view is the worst for single-image 3D.

## 10. Open problems, in priority order

1. **747 flap (the generalization test).** Run 8 targets, generic only: (a) teach acting on feedback when confident,
   using Glimmer's *own* r7 mistakes on training bodies (sample r7, keep its wrong answers, precise feedback, fix);
   (b) thin parts that look like real voxelized wings (ragged edges, uneven thickness, sheets mid-height not only on
   top); (c) hinge plates placed in a cleared channel, not over existing parts. Re-test on `r6-primitives`, then the
   747 untouched.
2. **Glimmer at large car size.** Running gear fails at `--size large` (trained on medium only; 43.2 mm wheels). Needs
   large-car rows.
3. **Camry manual** with the new glass: step search stuck on low rear plates; investigate `samgo/manual.py` ordering.
4. **GT1 stability**: cap door height (stop at the window line) or add a hinge layer.
5. **Route more through Spark**: glass region, door style (normal / scissor; true butterfly needs diagonal axes, not
   supported), wings/spoilers (currently glassed or blocky).
6. **1x1 plates** (~70-120 per car): the layout should choose bigger pieces; merging only helps pairwise.
7. Later, per the user: convertibles; "approved geometry" (known-good LEGO sub-assemblies) - plan before building.
8. User-only steps: rotate the Fireworks key, Studio check, video, submission.
