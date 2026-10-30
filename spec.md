# SAMgo

*LEGO comes from the Danish* leg godt, *"play well." SAMgo comes from* saml godt, *"assemble well," built on Meta's SAM.*

Photo → buildable LEGO-compatible model that **works**: a photo of a car becomes a brick car that rolls.

Meta Global AI Developer Hackathon · Indie track · Theme: Push the Frontier
Submission: **Wed Nov 4, 6:00 PM PT**

---

## 0. Current plan (frozen Oct 29; changes only by the user's decision)

This section is the single source of truth. `plan.md` (BrickGPT replication), `decoder.md` (shape-checked
decoding) and `mechanisms.md` (design discussion) are history; sections 1-6 below are the original spec, kept for
its stage results (S1-S6 are what step 1 here reuses).

### Pipeline
1. **Photo → body.** SAM 3.1 cuts out the object, SAM 3D makes the mesh, the algorithmic builder (S4) makes a
   faithful, buildable body. Spark's existing function spec (S2.5: "car, 4 wheels, 2 axles, rolls along x") says
   which tasks apply. Done (Camry at 20 and 60 studs).
2. **Glimmer makes it work, by editing the build.** Input: the spec, the region code supplies (wheel positions,
   window cells, door area) and the local parts. Output: edits (`set` / `add` / `remove`). Four tasks for Nov 4:
   1. real Technic running gear;
   2. glass windows (transparent parts);
   3. clean colors;
   4. preset doors (3821/3822, 3188/3189) on a small car.
3. **Code verifiers check every edit** (structure S3a, stability S3b, function S3c, door swing); errors go back to
   Glimmer, up to 3 retries.
4. **Demo:** photo → LEGO car that rolls, has glass and clean colors, and has doors that open, every mechanism
   placed by Glimmer and verified.

**Out:** Spark writing detailed instructions; Glimmer finding regions itself (views/zoom tools); built hinge doors,
ball joints, stability repair; v2. v1 (BrickGPT replication on Glimmer) finishes as a reported result only.

### Training data (generated, verified)
1. **Bases.** Vehicle meshes (Objaverse LVIS vehicles, no gating; our photo cars are held out for testing) are
   voxelized at ~16-24 studs and built by the car builder (`samgo/car.py`), giving complete builds whose running
   gear passes S3c. Only builds that pass are kept.
2. **One example per task per base.** Remove the mechanism from the complete build; the input is the build without
   it, the answer is the edits that restore it:
   - running gear: axles and wheels removed, Technic chassis bricks turned into plain bricks → edits that swap
     the Technic bricks back and add the axles and wheels;
   - glass: window cells as ordinary dark parts → recolor to transparent parts;
   - colors: noise added to the colors → edits back to the clean colors;
   - doors: closed side panel → cut the door gap and place preset door parts (a door-placement script, verified by
     the swing check).
3. **Format.** Chat rows: system = parts-and-rules sheet; user = spec + region + numbered local parts + task;
   assistant = the edit list. About 20% get a second turn (a flawed first fix, the verifier's error, the
   correction) so the model learns to use feedback.
4. **Every answer is checked** by applying it and running the verifiers; anything that fails is dropped.

### Training
- **SFT on base Glimmer** (Fireworks LoRA, free through Nov 30): run 1 running gear; run 2 + glass and colors;
  run 3 + preset doors. ~5-10k examples each, held-out test set fixed before training.
- **Evaluation** after each run on the held-out photo cars and held-out generated cars: pass rate on the first try
  and with 3 retries, against base Glimmer.
- **GRPO** only if SFT's first-try pass rate is 20-80% and retries don't already reach ~95%: a ~$10 test run, then
  ~100-300 steps, max $200. Reward: 0 if the edits don't parse or apply; else 0.5 × checks passed + 0.5 × match to
  the complete build. Details: `reports/SAMgo mechanism RL training plan.md`.

### Order of work
1. Running-gear data generator → SFT run 1 → evaluation.
2. Glass and colors in the same generator → SFT run 2.
3. Preset doors (parts, swing check, placement script) → SFT run 3.
4. Demo video.

---

## 1. Problem

Turning a real object into something you can physically build takes a designer: someone who can judge shape, pick parts, and make sure the result holds together. AI models can describe objects and generate pictures and meshes of them, but they are bad at producing **physically valid assemblies**: parts that exist, fit, connect, and stand up. The output looks right and falls apart.

LEGO is a clean testbed for this: a fixed part catalog, exact geometry, and validity that can be checked by a program. If a model can learn to go from a photo to a buildable LEGO design, the same recipe (perception → structured target → generation checked by a verifier) points at real manufacturing problems.

---

## 2. Solution (bird's eye)

1. **See it.** SAM 3D turns a photo of an object into a 3D shape. SAM 3.1 also segments its functional parts (wheels), giving their positions and sizes.
2. **Understand it.** Muse Spark infers what the object must *do*: "car → 4 wheels on 2 axles, rotating side to side."
3. **Define the target.** The shape becomes a LEGO-scale target: a grid of what should be filled, and in which color, plus the function spec.
4. **Build it.** A builder proposes a brick-by-brick design: a body for the shape, and a chassis, axles and wheels for the function.
5. **Check it.** A verifier, written as plain code, checks every design in three levels: **structure** (real parts, no overlaps, connected, grounded, on target), **stability** (force balance: it doesn't collapse), and **function** (axles turn in round holes, wheels clear the body, wheels are the lowest contact). Only designs that pass are shown or kept.
6. **Explain it.** Passing designs become step-by-step instructions and a parts list.
7. **Learn it (research).** An open-weight Meta model is trained to be the builder using only the verifier as its training signal (no human designs, no supervised warm start): expert iteration with a reverse curriculum over structural and functional tasks.

The product works without training (an algorithmic builder plus a Muse Spark agent, with the verifier in the loop). Two claims are new: **a build that works**, not just one that looks right; and **a model that learns to build from a verifier alone**. Prior photo-to-brick and learned brick generators produce static shapes and train on datasets of designs.

---

## 3. How we generate valid LEGO (bird's eye)

```
                         ┌──────────────────────────────────────────┐
  photo ──▶ SAM 3.1 ──▶  │  SAM 3D  ──▶  3D mesh  ──▶  voxelize     │
           (find/mask)   │  (shape)                    (LEGO scale, │
                         │                              palette)    │
                         └───────────────────┬──────────────────────┘
                                             ▼
                                       TARGET (grid)
                                             │
                                             ▼
                         ┌─────────────── BUILDER ───────────────┐
                         │  algorithmic (v0)  │  trained model   │
                         └───────────────────┬───────────────────┘
                                             ▼
                                       candidate build
                                             │
                                             ▼
                                ┌────────── VERIFIER ──────────┐
                                │ real parts? no overlaps?     │
                                │ connected? grounded?         │
                                │ matches target?              │
                                └───────┬──────────────┬───────┘
                                   fail │              │ pass
                                        ▼              ▼
                         repair / retry (agent)   VALID BUILD ──▶ instructions
                                        │                         + parts list
                                        └──────▶ back to BUILDER
```

**Training loop (research):**

```
   targets ──▶ model proposes K builds each ──▶ VERIFIER
      ▲                                            │
      │                                  keep only passing builds
      │                                            │
   new, slightly harder targets  ◀── fine-tune model on them
   (model proposes variants it can barely solve)
```

### Part model & build contract

- **Grid.** x, y in studs; z in plate heights (a brick is 3 plates). Targets use the same grid, so voxelization (S2) produces plate-height layers.
- **Palette: 22 rectangular parts.** Bricks and plates in 1×1, 1×2, 1×3, 1×4, 1×6, 1×8, 2×2, 2×3, 2×4, 2×6, 2×8. No tiles, slopes, or special parts.
- **Parts are written by size, not ID.** No rotation field: a 4×2 is a 2×4 turned sideways. LDraw IDs are used only for export (parts list, `.ldr`).
  ```json
  {"w": 2, "l": 4, "type": "plate", "x": 3, "y": 5, "z": 7, "color": "red"}
  ```
- **Colors.** 11 fixed LDraw colors. Color never affects validity, only target match.
- **Connected** means one part sits directly on another (top face meets bottom face) and they share at least one stud. Side contact doesn't count.
- **A build passes when:** every part is in the palette and in bounds, nothing overlaps, there is no floating part (all parts connect down to z = 0), everything is one structure, and it matches the target (IoU ≥ 0.85 and color accuracy ≥ 0.80 by default; curriculum tasks require exact occupancy). Not checked: clutch strength, balance, build order.
- **Curricula.** Two axes, raised one at a time: pieces missing (last 1, 2, 5, 10, … all) and part choice (size given → near-miss options → any of the 22). Any completion the verifier accepts counts, not only the original parts.

---

## 4. User flow (bird's eye)

```
  ┌────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Take/upload│──▶│ Confirm object  │──▶│ Choose size  │──▶│ 3D preview   │
  │ a photo    │   │ ("wooden chair" │   │ (small / med │   │ of the LEGO  │
  │            │   │  + mask shown)  │   │  / large)    │   │ model        │
  └────────────┘   └─────────────────┘   └──────────────┘   └──────┬───────┘
                                                                   │
                                                                   ▼
  ┌────────────┐   ┌─────────────────┐   ┌──────────────────────────────────┐
  │ Build it,  │◀──│ Download parts  │◀──│ Step-by-step instructions        │
  │ share it   │   │ list + LDraw    │   │ (one layer at a time, new bricks │
  │            │   │ file            │   │  highlighted)                    │
  └────────────┘   └─────────────────┘   └──────────────────────────────────┘
```

---

## 5. Schedule

| Day | Focus | Stages |
|---|---|---|
| **Wed 10/28** | Structural verifier; photo → target; algorithmic builder; functional parts + function verifier | S1, S2, S3a ✅, S4, Parts, S3c |
| **Thu 10/29** | Stability verifier; rolling-car builder; curriculum generator; GPU up; harness + baseline | S3b, S4f, S7 |
| **Fri 10/30** | One training run, structural + functional tasks mixed → **go/no-go** | S8 + S9 |
| **Sat 10/31** | More rounds; function spec; instructions | S8, S2.5, S5 |
| **Sun 11/1** | End-to-end app; trained model as alternate builder | S6 |
| **Mon 11/2** | Final evals, physical rolling car | S10 |
| **Tue 11/3** | Demo video, write-up, repo cleanup | — |
| **Wed 11/4** | Buffer · submit by 6 PM PT · post on X | — |

If S8 fails, the product (S1–S6) still ships and training is reported as a partial result. If behind, cut in order: hollowing (S2) → DPO round → per-brick stability scores (keep pass/fail). Protect the rolling car (S3c, S4f, S10) and S8 + S9.

---

## 6. Stages

Each stage proves exactly one thing. Each one depends only on stages before it.

| # | Proves | Done when |
|---|---|---|
| **S1** | A photo can become a usable 3D shape, plus where its functional parts are. | 5 of 5 test photos produce a recognizable mesh; for vehicles, SAM 3.1 also returns a mask per wheel. Masks are 2D and the target is 3D, so only **ratios** are used: wheel position along the car's length and diameter relative to it, mapped onto the mesh's longest horizontal axis; the far side comes from mirror symmetry. Requires a side-ish photo; if the two wheel masks differ in height by > 20% or the car's outline is implausibly short, fall back to wheels at 15% / 85% of length and flag it. ✅ 5/5 recognizable meshes (car, toy truck, chair, teapot, Beetle; a close-up hydrant photo failed). Wheel masks: one tight mask per visible wheel on 3/3 vehicles; a partly hidden far wheel also appears, so masks overlapping in x must be merged per axle. The 20% height guard misses a 3/4 view; wheel roundness (w/h) is the better signal. |
| **S2** | A 3D shape can become a LEGO-scale target that still looks like the object. | Renders are recognizable at the chosen size, without patchy colors (cluster colors first, fold rare ones). Large models are hollowed with internal pillars. ✅ chair, teapot, Beetle clearly recognizable; truck fair; sleek car loses detail at 20 studs. Dark reds snap to reddish brown (palette tuning pending). Hollowing not yet done. |
| **S2.5** | A Meta model can infer what an object must *do*. | Muse Spark returns the right function spec for 5 of 5 photos (e.g. `car → 4 wheels, 2 axles, side-to-side axis`). ✅ `samgo/function_spec.py` (Spark Open via Baseten, image in, JSON out): 6/6 photos get the right kind (3 vehicles, chair/teapot/hydrant static); cars 4 wheels / 2 axles; the dump truck 6 / 3 (plausible: a third, partly hidden wheel). The pipeline picks the builder from it. |
| **S3a** | We can tell a structurally valid build from an invalid one. | Verifier agrees with every hand-made good/bad test build (invalid part, out of bounds, overlap, floating, disconnected, off-target). ✅ `samgo/verifier.py`, 33 tests |
| **S3b** | We can tell a build that stands from one that collapses. | Force-balance check (linear program, scipy) agrees with hand-made stable/unstable cases, including one-stud overhangs; < 1 s under 300 pieces; per-brick scores. In training it runs only on samples that already passed S3a + S3c (checks in sequence, cheapest first); if that is still too slow, the reward uses a cheap support heuristic and full S3b runs at inference only. Reported "% stable" always uses full S3b. ✅ `samgo/stability.py`: per-piece force and torque balance, 4 rim contacts per stud, HiGHS LP; 8 hand-made cases (tower, 1- vs 2-stud overhang, tipping, floating, heavy arm, rolling car). 0.01 s mean, 0.66 s on a 665-piece car. Clutch 250 cell-weights per stud (≈1 N, low end of real) is an estimate to calibrate at S10. Procedural builds: 79/100 pass S3a+S3b when the builder stops at the first valid layout; ranking its 8 layouts by stud pull gives **88/100** (spec target 80). Photo cars: sports car (89/stud) and Beetle (66) stand; the truck's raised dump bed does not (2418): a tilted solid needs a hinge. |
| **S3c** | We can tell a car that rolls from one that doesn't. | Catches: axle through an axle-shaped hole (locked), wheel sweep blocked by the body, body lower than the wheels, misaligned axles. ✅ `samgo/technic.py`: also catches an axle with < 2 aligned holes, a part in the axle's path, a wheel off its axle, and < 2 wheeled axles; 9 tests. Geometry uses standard Technic dimensions (holes 5.8 mm up at stud edges, 30.4 mm wheel) as constants, to be confirmed. |
| **Parts** | Functional parts fit the stud grid. | Technic brick with round hole, Technic brick with axle hole (locks the axle; included so the verifier must catch it), axle, and wheel+tire verify correctly. The bricks sit on the stud grid, but **holes sit at half-stud offsets** (stud-edge coordinates) about 5.8 mm above the brick's bottom, so functional geometry (holes, axles, wheels) is in millimeters while structure stays in studs and plates. The chassis layer fixes wheel center height; ground clearance = wheel radius − (hole height + body bottom above the chassis bottom) must be > 0. For vehicles, "grounded" means connected (studs, axles in holes, wheels on axles) to a wheel touching the ground. |
| **S4** | An algorithm can turn a target into a valid, stable build. | ≥ 80% of procedural targets pass S3a + S3b. Greedy layer tiling (seam and support scoring), then repair: recombine → bridge → remove, weakest bricks first. ✅ S3a: 93.8% of 500 (`samgo/builder.py`); stability-aware repair pending S3b |
| **S4f** | An algorithm can build a car that rolls. | A car target passes S3c: round-hole chassis at the wheel positions, axles, wheels, body on top. ✅ `samgo/car.py`: the S1 sports-car photo's target passes S3a and S3c (396 parts, IoU 0.975, whole shape kept by lifting it onto the chassis layer). Wheel positions come from SAM 3.1 masks as ratios (`samgo/wheels.py`), else 15% / 85%. |
| **S5** | A valid build can become instructions a person can follow. | Someone builds a small model from the generated steps without help. Generator done (`samgo/instructions.py`): layer-by-layer steps with 3D renders (new parts in color, earlier ones faded), per-step parts and a bill of materials; the sports car is 47 steps. Not yet tested on a person. |
| **S6** | A user can go from photo to a working build with no manual steps. | A new car photo goes through the app end to end, with the verifier rejecting and rolling back bad steps. CLI done (`samgo/pipeline.py`): photo → function spec → SAM 3.1 + SAM 3D → target → wheels → verified build → instructions, no manual steps. Sports car, Beetle, truck: all pass structure and function; car and Beetle stand, the truck's raised bed does not. Chair: valid structure, does not stand. Verifier-feedback rollback with the LLM builder not wired in yet; no app UI yet. |
| **S7** | An open model can attempt builds, and we can measure how well it does. | Baseline on a frozen set of 200+ targets: % valid, % stable, IoU, rollbacks per build. Piece encoding (absolute vs. relative to parent) decided. |
| **S8** | Training against the verifier makes the model better at building. | **Trained model ≥ +5 points absolute full-build pass rate over baseline, or ≥ 30% fewer rollbacks per build at an equal or higher pass rate.** Gate frozen before training: same verifier version, decoding settings, and rollback budget. Expert iteration: K samples per target, cap kept per target, fresh targets every round; each round fine-tunes the base model on **that round's samples only** (not accumulated), capped at ~5k, so per-round training time stays flat. *Results so far (Oct 28):* expert-iteration SFT on Glimmer-30B was NO-GO twice (94 and 692 examples: −9 and 0 points on trained levels); one offline round only reinforces answers the model already gives. Online GRPO (`pilot/grpo.py`) ran 3 updates before Modal credit ran out: flat, uninformative. Verifier *feedback* at inference works: Spark Open from empty 0.35 → 0.72 in 4 attempts vs 0.56 for best-of-4. |
| **S9** | The model learns function, not just shape. | In the same run as S8, functional tasks (put back a car's wheels and axles) are mixed into the curriculum; the trained model's functional pass rate beats baseline. |
| **S10** | It's real. | A brick car built from the instructions rolls across a table, next to the original object. |

S1–S6 = the product. S7–S9 = the research. S10 = the demo.

**Curriculum (S8 + S9).** Two task families, structural (fill a target) and functional (wheels, axles), each with "pieces missing" levels from the last 1 up to all. Every batch mixes levels, weighted toward those where the model succeeds 10–90% of the time; levels advance on measured success, not on a fixed schedule. Out of scope: steering, gears, other mechanisms.

---

## 7. Dependencies

### API keys & accounts

| Service | Env var | Used for | Needed by | Status |
|---|---|---|---|---|
| Meta Model API | `MODEL_API_KEY` | Muse Spark (agent), SAM 3.1 (masks). $150 credits; card must be on file; claim by Nov 4 | S1 | ☐ |
| fal.ai | `FAL_KEY` | SAM 3D Objects (photo → mesh), ~$0.02/call | S1 | ☐ |
| Hugging Face | `HF_TOKEN` | Download open-weight model (e.g. Muse Glimmer 30B) | S7 | ☐ |
| Baseten | `BASETEN_API_KEY` | Muse Spark Open baseline (`meta-models/muse-spark-open-1026`) | S7 | ☐ |
Keep keys in `.env` (gitignored). The repo is public.

### Software

| Area | Packages / tools |
|---|---|
| Core | Python 3.11+, numpy, Pillow |
| Mesh & voxels | trimesh (GLB loading, voxelization) |
| Model APIs | openai SDK (Meta Model API is OpenAI-compatible), fal-client, anthropic |
| Training | vLLM (sampling), transformers, peft (LoRA/QLoRA), bitsandbytes, accelerate |
| Rendering | LDraw parts library (check its license before bundling), renderer (LDView or a custom voxel renderer) |
| App | Gradio |

### Data

| Item | Source | Needed by |
|---|---|---|
| Test photos | Own photos of chunky, matte objects | S1 |
| Synthetic targets | Procedural shapes + licensed meshes only (hackathon bans assets without rights) | S4 / S7 |
| Held-out eval set | Generated once, frozen, never trained on | S7 |

### Physical

| Item | Needed by |
|---|---|
| LEGO bricks for the demo build (order early or use what's on hand) | S10 |
