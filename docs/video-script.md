# SAMgo: launch film (3:00)

A product launch, not a research talk: short lines, confident pace, every cut on a beat. Your voice throughout,
recorded beat by beat (one file per section: `01_promise.m4a`, `02_different.m4a`, ...); the cuts are timed to your
words with whisper word timings. Asset paths: `video_handoff.md` section 3.

**Look.** One system for the whole film: near-black backgrounds between shots, off-white Futura Medium for every word
on screen (the SAMGO title's type), one accent colour, LEGO red, used only for the checker's corrections and the
manual's arrows. Lower-thirds name each Meta model as it works. Kinetic type: words cut on with the voice, never
fade in slowly.

**Sound.** One track that builds from a pulse to a full drop at the build-together moment; the boom on the title; a
real brick click on every stop-motion frame and every part that lands; a soft whoosh on each pipeline morph.

---

## 0:00-0:10 · Cold open (on camera)

**You:**
> We know models can draw.
> But can they build, in the real world?
> Introducing SAMgo.

**Picture:** you, landscape. On the vowel of "Sam", **SAMGO** cuts on, huge, off-white, with the boom (the "QUEENS"
style, `scripts/title_card.py`). Hold 2 s. Smash cut to black.

---

## 0:10-0:24 · The promise

**VO:**
> SAMgo turns a single photo into a LEGO model you can actually build.
> Real doors. Real wheels. Real pieces.
> At any size.

**Picture:** the red Camry photo (`steps/1_photo.png`), slow push-in. On "LEGO model", match-cut to the LEGO Camry at
the same angle (`clips/01_Camry.mp4`). "Real doors" - the doors swing. "Real wheels" - the wheels turn. "Real pieces"
- the model bursts apart into its parts, then snaps back. "At any size" - three hard cuts through the resolution
sweep, small to large.

---

## 0:24-0:46 · Not image generation

**VO:**
> This isn't image generation.
> Every piece comes from the real LEGO catalog. Every connection is checked, with exact math, so the model holds
> together, piece by piece.
> And it doesn't just look like the machine. It works like it.
> Rotors that spin. Engine fans that turn. Doors that open.

**OST (kinetic, with the voice):** REAL PARTS · CHECKED CONNECTIONS · WORKING MECHANISMS

**Picture:** "real LEGO catalog" - a grid of the model's actual parts filling the frame. "Checked, with exact math" -
close-up of studs; a checker sentence types on in red, flips to green. "Rotors / fans / doors" - three hard cuts on the
beat: helicopter rotor, 747 engine fan, Camry door.

---

## 0:46-1:30 · How it works (the Camry)

Each stage morphs into the next: the photo's mask becomes the mesh, the mesh dissolves into cells, the cells snap
into bricks. A thin progress rail along the bottom marks each stage as it lights up.

**VO:**
> It starts with Meta's SAM 3.1. It finds the car, and exactly which parts of it are glass.
> SAM 3D lifts that one photo into a full 3D shape.
> We turn the shape into LEGO, real plates and bricks, layer by layer.
> Then Muse Spark looks at the car the way you would. It knows what moves: these are wheels, these are doors, and
> this is where they go.
> Muse Glimmer, which we fine-tuned, builds them. Every answer goes through a checker. When something's wrong, it
> says exactly what, and Glimmer fixes it.
> And then, it all comes together.

**Picture, in order:**
1. `1_photo.png` -> `2_sam31_mask.png` (the mask wipes across) -> `3_sam31_glass.png` / `overlay_windshield.png`
   (glass glows). Lower-third: **SAM 3.1 · Meta**.
2. `4_sam3d_mesh.mp4`, the mesh turning, the windshield marked. **SAM 3D · Meta**.
3. `5_voxels.png` -> the resolution sweep settling on the built body.
4. `6_spark_layout_side.png`: Spark's points land on each wheel and door, one per beat. **Muse Spark · Meta**.
5. The feedback moment (`docs/feedback-example.md`), styled as a terminal: Glimmer's answer, the checker's line typed
   in red ("these hinge parts hold nothing..."), the fix, the line turning green. **Muse Glimmer · Meta, fine-tuned
   by us**.
6. The drop: `clips/02_Camry_build.mp4`, chassis, axles sliding through their holes, the body rising layer by layer,
   faster and faster into one car.

---

## 1:30-1:50 · The manual (stop motion)

**VO:**
> And it comes with instructions.
> Fifty-six steps, each one proven stable before the next piece goes on.
> Plus the exact parts list, ready to order.

**Picture:** the 56 step pictures (`runs/build_kit/camry_red_current/steps/step_*.png`) as stop motion with a click
per frame: slow for the first few, then accelerating, so the car grows in front of you; land on `finished.png`; slide
in the parts list page. **OST:** 320 PIECES · 56 STEPS

---

## 1:50-2:14 · It generalizes (montage)

**VO:**
> And it's not just cars.
> SAMgo never saw any of these machines in training. It learned general skills on simple blocks, hinge an edge,
> spin a rotor, turn a turntable, and it uses them on machines it has never seen.

**Picture:** `clips/montage.mp4` re-cut to the beat: 747 fans and flaps, helicopter rotors, Cessna propeller and
ailerons, supercar; end on the wind turbine turning, slow. **OST:** NONE OF THESE WERE IN THE TRAINING DATA

---

## 2:14-2:44 · Why it matters (on camera, then over footage)

**You:**
> Here's why this matters.
> Manufacturing is about assembling known parts. But figuring out how they fit together, and in what order, is still
> work people do by hand.
> SAMgo shows a model can do it: pick real parts, make the mechanisms work, and plan an assembly that holds at every
> step.
> Today, the testbed is LEGO. Next, it's real CAD parts, and finding the best way to build them.

**Picture:** you, two lines to camera; then over your voice: parts list -> certified steps -> a mechanism sweeping
clear -> the Camry's `model.ldr` in BrickLink Studio. On "Next, it's real CAD parts", a slow push-in to black.

---

## 2:44-2:54 · The numbers

**OST, one per beat, big, music up:**
- 50 / 50 held-out cars
- 10 / 10 doors on vehicles it never saw
- one training run: upright rotors 8 -> 24 / 30

---

## 2:54-3:00 · End card

**SAMGO** (the title type) · github.com/vvennela/samgo
Built with SAM 3.1 · SAM 3D · Muse Spark · Muse Glimmer
team · photo credits

---

## Recording notes

- One file per section, quiet room, phone close (Voice Memos is fine). A second of silence before and after each take.
- Read slower than feels natural, and land the short lines ("Real doors. Real wheels. Real pieces.") with a beat
  between them: the cuts hit on those beats.
- Small wording changes are fine: the cut follows your words, not this page.
- Civil machines only on screen.
