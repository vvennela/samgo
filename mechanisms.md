# Mechanisms: the model makes the build *work*

Status: **design for review (Oct 29)**. Nothing here is built yet. It replaces "the model builds the shape" (decoder.md),
after the chair benchmark showed a text-only model cannot be steered into *your* object (8-13% coverage, towers),
while the algorithmic builder gets 98-99.9% shape match.

## The split

| Step | Who | Output |
|---|---|---|
| 1. Photo -> object | SAM 3.1 (mask), SAM 3D (mesh), Spark (what it is, what it must do) | mesh, `function_spec` |
| 2. Shape -> buildable body | **algorithm** (S4 builder) | faithful, stable brick body |
| 3. Body -> working object | **Glimmer** | mechanisms: running gear, hinged doors, ... as part edits |
| 4. Check | **code verifiers** (exact physics) + **Spark** (does it look right) | pass/fail + error text fed back to Glimmer |

The model's job is the part a camera cannot see and an algorithm cannot guess: where the axles go, how big the wheels
are, where a door's hinge line runs, and how to change the body so it works. That is mechanism design, and every
"it works" claim is checked by a verifier, not asserted.

## Demo target (Nov 4)

A photo of a Camry becomes a LEGO Camry that **rolls** and whose **doors open**, with Glimmer placing the axles, wheels
and hinges, and the verifiers proving each one works. Shown as a build-up video (see the viewer in decoder.md).

## Mechanism 1: running gear (axles and wheels)

What exists: `samgo/car.py` places running gear algorithmically; `samgo/technic.py::verify_function` checks it.
Where the algorithm fails, which is where the model earns its place:
- wheel positions are guessed (15% / 85% of the length) when the photo is not a side view;
- one through-axle per wheel pair, so cars wider than a 16-stud axle are refused (the 60-stud Camry);
- one wheel size (30.4 mm), too small for large models.

**Model task.** Input: the body (parts), the car's dimensions, Spark's spec ("car, 4 wheels, 2 axles, rolls along its
length"). Output: edits — Technic bricks for axle holes, axles, wheels (size from a list), and removals where the body
must make room (wheel arches).

**Verifier (extend `technic.py`).**
- Stub axles: one short axle per wheel through 2 aligned round holes; wheels on the same line count as one wheel pair.
- Wheel sizes: a small list of real LEGO wheel/tyre diameters, each with its clearance radius.
- Existing checks: axles turn (round holes, not cross holes), wheels clear the body while turning, wheels are the
  lowest contact, at least 2 parallel wheeled axles, the car stands on its wheels.

## Mechanism 2: doors with hinges

**Model task.** Find each door in the body (window line, pillars; Spark can point at door seams in the photo), cut the
door panel free, and place hinges on its front edge so it swings out.

**Verifier (new).**
- The door is its own connected group, attached to the body only through hinges.
- All hinges on a door share one pivot axis.
- Sweep the door from closed to ~70 degrees open in steps: no collision with the body at any step.
- Stability passes with the door closed and open.

**New parts.** Hinges are not in the palette yet. Add one or two common hinge types with LDraw IDs (to verify against
the official library, as for the 35-part palette) and their pivot geometry.

## Encoding LEGO knowledge in the model

Three layers, cheapest first; each is measured before moving to the next.

1. **Parts-and-rules library in the prompt.** Every functional part with its connection rules (an axle turns in round
   holes and locks in cross holes; a wheel needs clearance of its radius; a hinge's halves rotate about its pin) and
   2-3 worked examples. Tells us how far base Glimmer gets without training.
2. **SFT on generated, verified mechanisms.** The builders and verifiers can generate unlimited *correct* examples:
   take an algorithmic car body, remove its running gear, and the answer is running gear that passes the verifier;
   same for doors. Free under the Fireworks SFT promotion (through Nov 30), same trainer as v1. Feedback turns
   (verifier error -> corrected edit) are included so the model learns to use the feedback loop.
3. **Real LEGO designs (stretch).** The LDraw official model library holds real sets, including Technic, as part
   lists. Mining how real models use hinges and axles adds human-designed examples. Check its size and contents first.

## Feedback loop

Per edit: the code verifier returns exact errors ("axle 3 runs through a cross hole, so it cannot turn"; "door 1 hits
the B-pillar at 35 degrees"). After a full attempt: Spark compares the render with the photo. Glimmer continues with
both in its context until every verifier passes, or a retry cap is hit (reported honestly). Pilot evidence: verifier
feedback retries beat best-of-k (0.72 vs 0.56 from empty).

## Measuring it

Frozen evaluation set: the existing photos (car, beetle, truck, Camry) plus held-out generated bodies. Per mechanism:
pass rate of the verifier, retries needed, calls and time. Compared: base Glimmer with layer 1, then the layer-2 model.

## Plan

1. Running gear: stub axles + wheel sizes in the verifier; the model task and prompt library; test base Glimmer on
   the four car photos.
2. If base Glimmer falls short: generate running-gear SFT data and train (slot is free after v1).
3. Doors: hinge parts, the swing verifier, the model task; same test-then-train path.
4. Viewer and launch video.

## Open decisions

- Model size for the demo: 20 studs (works today) or ~40-60 studs (more recognizable; needs stub axles and larger
  wheels).
- Which hinge parts to add.
- Training: v1 finishes ~9:30 PM; v2 (35-part palette from empty) is cancelled to keep the slot for mechanism SFT.
