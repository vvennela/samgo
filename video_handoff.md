# Video handoff (Meta hackathon demo)

Everything needed to cut the 3-minute video: rules, story, the assets that exist (with paths), how to make more, the
numbers for the screen, and how to present them. State as of **Nov 4, ~01:30 PT**.

## 1. Hard requirements (from the hackathon rules)

- **Due Wed Nov 4, 6:00 PM PT** (9:00 PM ET) on Cerebral Valley, with the **public** repo link (check it is public)
  and everyone who worked on it listed.
- **3 minutes max.** Judges watch the video in the live round (top 5 per track): it is the main artifact.
- **Only work done during the hackathon (Oct 27 - Nov 4).** All of this repo qualifies (history starts Oct 27).
- **Show real use of Meta models:** SAM 3.1 (Meta Model API), SAM 3D, Muse Spark Open (Baseten), Muse Glimmer
  (fine-tuned by us on Fireworks). Name them on screen.
- **No military objects.** Never show the F-16 (`photos/f16.jpg`) or anything from it. Civil machines only.
- **Only assets we have rights to** (photo sources: `photos/SOURCES.txt`).

## 2. The story

**The full polished script, word for word with timings: `docs/video-script.md`.**

One object end to end, then breadth, then the proof it is real: the generated LEGO-style manual shows every step
of the build.

| Time | Shot | Asset |
|---|---|---|
| 0:00-0:10 | **"SAMgo"** in a huge bold sans (the "QUEENS" title style) over SF / San Jose skyline footage | stock footage (Pexels / Pixabay), text |
| 0:10-0:20 | Your hand holds a phone showing the Camry photo; lower the phone -> cut to the LEGO car, doors swinging, wheels turning | `runs/video/steps/1_photo.png`, `runs/video/clips/01_Camry.mp4` |
| 0:20-1:20 | **Every step, visualized** (voice over each): SAM 3.1 mask and glass -> SAM 3D mesh -> voxels -> Spark names the working parts -> Glimmer builds them, the checker's sentence, the fix -> the build animation | `runs/video/steps/*` (table in section 3), `docs/feedback-example.md` as on-screen text, `runs/video/clips/02_Camry_build.mp4` |
| 1:20-1:45 | **You can build it:** flip through the LEGO-style manual (callouts, arrows, parts list) | `runs/build_kit/camry_red_current/Camry_manual.pdf` |
| 1:45-2:15 | **It generalizes:** the montage (747 fans + flaps, helicopter rotors, Cessna propeller + ailerons, supercar) with one line: "none of these was in the training data" | `runs/video/clips/montage.mp4` |
| 2:15-2:40 | **Numbers** (one slide, section 5) | text |
| 2:40-3:00 | You on camera: why it matters ("push the frontier": machines in the physical world), credits | camera |

More detail: `docs/demo-script.md`, `demo.md`.

## 3. Assets that exist

**Clips** (`runs/video/clips/`, 1920x1080, 30 fps; recorded from the 3D viewer; descriptions in `runs/video/README.md`):

| file | length | what |
|---|---|---|
| `01_Camry.mp4` | 8 s | red Camry orbiting, doors swinging, wheels turning |
| `02_Camry_build.mp4` | 14 s | the build animation: chassis first, body bottom-up, speeding up |
| `03_Supercar.mp4` | 8 s | blue supercar |
| `04_747.mp4` | 8 s | 747: four fans spinning, flaps + ailerons, gear |
| `05_Helicopter.mp4` | 8 s | main rotor + tail rotor spinning |
| `06_Cessna.mp4` | 8 s | propeller spinning, control surfaces moving |
| `montage.mp4` | 30 s | 03-06 back to back |

**Pipeline stills, Camry** (`runs/video/steps/`): `1_photo.png`, `2_sam31_mask.png`, `3_sam31_glass.png`,
`overlay_windshield.png`, `overlay_car_windows.png`, `3b_glass_marked_photo.png`, `4_sam3d_mesh.mp4` (6 s),
`5_voxels.png`, `6_spark_layout_side.png`.

**Interactive pages** (screen-record them; "Preview motion" moves every joint, "Build it" plays the build):
- Showcase (Camry, supercar, 747, helicopter, Cessna; updated Nov 4 with fewer pieces):
  https://claude.ai/artifact/WDqu8phpQUANxebg8BWncL
- Red Camry alone: https://claude.ai/artifact/GDPzA6XN37pSiKQ4Ya4ssa
- Resolution sweep (voxels 1x to 10x): https://claude.ai/artifact/MCUrJq8XbVmcDtatBsuZPy

**Build kit / manual** (`runs/build_kit/camry_red_current/`, newest): 320 parts, **56 certified steps**,
`Camry_manual.pdf` (60 pages, LEGO style: parts callout per step with counts, names, colour and part number; red
arrows; studs and black outlines; parts list by colour at the end), `parts.csv`, `bricklink_wanted.xml`, `model.ldr`
(opens in BrickLink Studio). Use this kit's numbers on screen (320 parts, 56 steps).

**Feedback example** (`docs/feedback-example.md`): a real case, word for word: Glimmer's first answer, the checker's
sentence ("these hinge parts hold nothing..."), the fix that passes. Good as on-screen text.

## 4. Making new footage

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/samgo
# a viewer page for any build(s)
PYTHONPATH=. ~/.venvs/fwcookbook/bin/python -m samgo.viewer3d out.html "Name=path/to/build.json" ...
# a clip: seconds, optional ":build" for the build animation, optional camera distance
PYTHONPATH=. python scripts/record_video.py out_dir "Name=path/build.json:8" "Name=path/build.json:14:build"
# manual pages as images (for a flip-through)
pdftoppm -r 150 -png runs/build_kit/camry_red_current/Camry_manual.pdf out_dir/page
```

Models (builds as JSON; current = re-merged with current LEGO sizes, Nov 4):
`runs/hand_finished/{camry,gt1,747,helicopter,cessna}_current.json`.
A purely automatic one worth a clip (no hand-finishing at all, passes every check, stands, rotor turns):
the 40-stud wind turbine, `runs/demo_r11_xl/wind_turbine/artifacts/*/build.json`.

## 5. Numbers for the screen

All from frozen tests (made before training, never regenerated, never trained on); run 11 is the current model.
Sources: `fireworks-training-runs/objects-r11/run.md`, `objects-r10/run.md`.

- **112 frozen test cases**: 92 built on the first try (run 11).
- **Cars, 50 problems**: 50/50 within the feedback rounds (48 first try). **Doors on unseen vehicles**: 10/10.
- **New skills from one generic training run each**: side rotors 1 -> 30 of 30 (run 10); upright rotors 8 -> 24 of
  30 (run 11).
- **Feedback works**: retries with the checker's sentences beat sampling more answers (.72 vs .56 from empty).
- **At 40 studs** run 11 built every rotor it was asked for on the first try.
- **Physical checks**: every part a real LEGO element (LDraw geometry within 0.4 mm), held by studs, mechanisms
  sweep clear, stands under a force balance, build order certified step by step (56 steps for the Camry).
- Run 13 (the same skills on bodies 1.5-2.5x larger) finishes ~06:00; its results go in
  `fireworks-training-runs/objects-r13/run.md`.

## 6. How to present it

- None of the objects is in the training data: Glimmer trains on generic bodies with no object words; the 747,
  Cessna, helicopter and cars are held-out exams.
- The showcase models are the pipeline's builds completed with SAMgo's own mechanism builders, every one verified by
  the same checker (the viewer labels them). The fan and the 40-stud wind turbine come straight from the pipeline.
- The build animation's paths are proved collision-free in Lean (`formal/AnimationPath.lean`); buildability is
  certified step by step with exact arithmetic (`samgo/manual.py`).
- Closing line: bigger models are next.

## 7. Before submitting

1. Credits slide: team members; models (SAM 3.1, SAM 3D, Muse Spark, Muse Glimmer); photo sources; any stock
   footage URL.
2. Push, and check the repo is public with no keys in it (`.env` is ignored).
3. Optional: re-record `01_Camry.mp4` from `camry_current.json` so the clip matches the 320-part manual.
