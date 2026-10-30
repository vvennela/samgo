> **Superseded (Oct 29):** history only. The current plan is `spec.md` (repo root) section 0.

# Decoder proposal: building *your* object, not just *an* object

Status: **decided Oct 29 and built** (`brickgpt/shape.py`, `brickgpt/decode.py`, `brickgpt/build_photo.py`).

## Decisions (Oct 29)

| Choice | Now | Later |
|---|---|---|
| Parts outside the object | **Hard reject**, 1-cell tolerance at the edges | tune the tolerance from real outputs (`--tolerance`) |
| Stopping early | refused below **70%** coverage | raise to 90% once builds reliably reach it (`--min-coverage`) |
| Best-of-k per part | **on, k = 3**: one call per part returns 3 candidates from the same partial build; keep the one filling the most still-empty target cells | compare with k = 1 on the same photos |
| Color | paint each part afterwards from the photo | |
| Order | this decoder first; shape-conditioned training only if fidelity is not good enough | |

Best-of-k is per part, inside one build: the prompt and partial build are sent once per call for all k
candidates, so the extra cost is k x ~10 sampled tokens per part (~1.3-2x), not k x the whole build. It does
use one call per part (no 8-lines-per-call speedup), and greedy coverage can favor big parts; the stability
rollback catches structural damage from that.

## The gap

The BrickGPT recipe (v1, training now) conditions on text only. A photo becomes a caption, and Glimmer builds
*a* chair matching it, not *your* chair: proportions, seat height and back shape come from the model's prior,
not from the photo. The model never sees the SAM 3D shape.

## Idea

Keep Glimmer as the builder, and make the verifier check every brick against your object as it is placed.
The model proposes; the verifier rejects bricks that leave your object's shape, refuses to stop before the
shape is covered, and (optionally) picks, among a few bricks the model proposed, the one that fills your
object best. Every placement is still the model's.

Inputs per build:
- **caption**: Muse Spark Open's detailed description of the photo (as BrickGPT's prompt);
- **target**: the SAM 3.1 + SAM 3D shape, voxelized onto BrickGPT's 20 x 20 x 20 grid (brick heights);
- **colors**: the photo's per-voxel colors from S2.

## Options

| # | Option | What it does | Cost | Proposed |
|---|---|---|---|---|
| 1 | Inside-the-shape check | Reject and resample a brick that fills cells outside the target (BrickGPT's collision rule, extended), with a 1-cell tolerance at the edges | extra calls only when the model strays | **default** |
| 2 | No early stopping | Refuse the model's end of answer until coverage of the target reaches a threshold (70% now, 90% later); give up honestly after a cap | a few calls | **default** |
| 3 | Best-of-k per brick | Ask for k candidates per brick (one call, `n=k`), keep the one that fills the most still-empty target cells | one call per part, ~1.3-2x tokens | **default, k=3** |
| 4 | Best-of-N whole builds | Build N models, keep the best match | N x time | rejected |
| 5 | Stability rollback | Cut to the first unstable brick and continue (BrickGPT) | as now | **default** |
| 6 | Color from the photo | Each brick takes the majority color of the target cells it covers | free | **default** |
| 7 | Format masking | Constrain sampling to well-formed brick lines (BrickGPT's logit masking) | free if Fireworks grammar mode works on raw completions | if easy |

## Design questions

- **Scale**: S2 targets are 20 studs along the longest side in plate heights; convert to brick heights (z / 3)
  to match the grid v1 learned.
- **Tolerance**: thin parts (chair legs, slats) are one cell wide; allow +-1 cell at the target's edges so
  near misses are not rejected.
- **Stuck regions**: temperature rises on repeated rejection (BrickGPT); after a cap, stop and report the
  uncovered region. Code never places bricks for the model.
- **Visibility**: every reject ("outside your chair"), forced continuation and rollback is an event, so the
  live demo view shows the model being corrected.

## Measuring "is it my chair?"

Reported on every build:
- **Coverage**: fraction of target cells filled (70% required now, aim 90%).
- **Precision**: fraction of built cells inside the target, with the 1-cell tolerance.
- **IoU** with the target (aim >= 0.7 as "clearly your object").
- **Image similarity**: render the build and compare it with the photo using CLIP and DINO (BrickGPT's
  metrics), to catch right-volume-wrong-look.

## Detail (resolution / piece count)

From the user's side one knob, "how detailed": more pieces = more detail.
- v1: approximated by the decoder's brick budget, with best-of-k preferring bigger bricks when the budget is tight.
- v2: learned, by adding "Use about N parts." (true N) to the v2 training prompts. Needs a rewrite of the v2
  file before v2 starts (minutes, no new data).
- Model size (studs across) as a knob is a **stretch goal**: it needs training on several grid sizes, or
  building large models in 20 x 20 x 20 windows, both of which need the shape-conditioned model below.

## Later: shape-conditioned training

Put a compact encoding of the target shape in the prompt (as BrickAnything conditions on point clouds), so the
model proposes the right bricks from the start instead of being steered: fewer rejections, faster builds, and
the route to resolution as an input and windowed large builds. A separate training run after v1.

## First real check (Oct 29, base Glimmer, chair, $0.70 cap)

Plumbing worked (595 calls, best-of-3, budget guard). The build failed: 75% of proposals fell outside the
chair, the model re-proposed the same rejected brick up to 54 times, and the end-of-build stability check found
a floating brick at #5 and rolled back 71 bricks. Fixes since: rejected lines are remembered per position
(repeats raise temperature +0.05 and 40 repeats stop the build as stuck), and floating parts are rejected when
placed. The outside-shape rate is the text-only model's own idea of "a chair": the real fix is giving the model
the shape (shape-conditioned training), now the likely next training run after v1/v2.

## Still open

- Add the part count to the v2 prompts (before v2 trains)?
