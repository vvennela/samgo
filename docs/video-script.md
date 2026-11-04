# SAMgo: launch film (3:00)

A product launch: every cut on a beat. **The narration is yours, in your words** (below, as you said it; the only
edits are three optional word swaps marked [like this], each a claim judges could check). Recorded beat by beat (one file per section: `01_promise.m4a`, `02_different.m4a`, ...); the cuts are timed to your
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
> We know models can draw, but can they build in the real world? Introducing SAMgo.

**Picture:** you, landscape. On the vowel of "Sam", **SAMGO** cuts on, huge, off-white, with the boom (the "QUEENS"
style, `scripts/title_card.py`). Hold 2 s. Smash cut to black.

---

## 0:10-0:24 · The promise

**You:**
> SAMgo takes any photo of an object and turns it into a LEGO model you can really build. Yes, complete with doors and
> tires, using real pieces, at any piece count.

**Picture:** the red Camry photo (`steps/1_photo.png`), slow push-in. On "LEGO model", match-cut to the LEGO Camry at
the same angle (`clips/01_Camry.mp4`). "Doors and tires" - the doors swing, the wheels turn. "Real pieces" - the model
bursts apart into its parts and snaps back. "Any piece count" - hard cuts through the resolution sweep, small to large.

---

## 0:24-0:46 · Not image generation

**You:**
> This is completely different from image generation, because SAMgo builds with real pieces from a catalog, and
> rigorously [checks with exact math] that each piece actually [connects] to another. Then, it also builds the relevant
> mechanism! So for helicopters, it builds the rotor, for planes, [engine fans], and for cars, doors and wheels.

Swaps: "formally verifies" -> "checks with exact math" (the build is certified with exact arithmetic; Lean proves the
merge rule and animation paths, not the whole build); "molds" -> "connects" (pieces click on studs); "turbines" ->
"engine fans" (what the 747 shows).

**Picture:** "real pieces from a catalog" - a grid of the model's actual parts. "Each piece connects" - close-up of
studs, a checker sentence types on in red, flips to green. "Helicopters / planes / cars" - three hard cuts on the beat:
helicopter rotor, 747 engine fan, Camry door.

---

## 0:46-1:30 · The pipeline (the Camry): your own words

Say it however it comes out naturally, one or two lines per step. What each shot shows, so you know what to name:

1. **SAM 3.1** (Meta) finds the car in the photo and which pixels are glass: `1_photo.png` -> `2_sam31_mask.png` ->
   `3_sam31_glass.png` / `overlay_windshield.png`.
2. **SAM 3D** (Meta) turns the one photo into a 3D mesh: `4_sam3d_mesh.mp4`.
3. **Our code** turns the mesh into LEGO cells, then real plates and bricks: `5_voxels.png`, the resolution sweep.
4. **Muse Spark** (Meta) knows what moves and where: wheels, doors: `6_spark_layout_side.png`.
5. **Muse Glimmer** (Meta, fine-tuned by you) builds them; the checker says exactly what's wrong; Glimmer fixes it:
   `docs/feedback-example.md` as a terminal, red line -> green.
6. The build comes together: `clips/02_Camry_build.mp4`.

Each stage morphs into the next, a progress rail along the bottom, a lower-third naming each model.

---

## 1:30-1:50 · The manual (stop motion)

**You:** (your words; the facts: 56 steps, each one checked stable, a parts list you can order)

**Picture:** the 56 step pictures (`runs/build_kit/camry_red_current/steps/step_*.png`) as stop motion with a click
per frame: slow for the first few, then accelerating, so the car grows in front of you; land on `finished.png`; slide
in the parts list page. **OST:** 320 PIECES · 56 STEPS

---

## 1:50-2:14 · The montage

**You:** (optional; the fact: none of these machines was in the training data)

**Picture:** `clips/montage.mp4` re-cut to the beat: 747 fans and flaps, helicopter rotors, Cessna propeller and
ailerons, supercar; end on the wind turbine turning. **OST:** NONE OF THESE WERE IN THE TRAINING DATA

---

## 2:14-2:44 · Why it matters (pushing the frontier)

**You:**
> A lot of manufacturing use cases are assembling from known parts, but humans still have to figure out the [assembly].
> Imagine instead if a model could figure [that out]? SAMgo does it, with LEGO as a testbed. In the future, to improve
> on SAMgo, I would use it on real-life CAD [and find the best assembly].

Swap: SAMgo finds an assembly that holds at every step, not the optimal one, so "optimal/best" moves to the future
line.

**Picture:** on camera for the first line; then parts list -> certified steps -> a mechanism sweeping clear -> the
Camry's `model.ldr` in BrickLink Studio; on "real-life CAD", a slow push-in to black.

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
