# SAMgo: the 3-minute script

Voice-over (VO) is about 400 words at a calm 140-150 words a minute; on-screen text (OST) is short and big. Every
asset path is in `video_handoff.md` section 3. Name the Meta models on screen as they appear.

---

## 0:00-0:09 · Cold open (on camera)

**Picture:** you, landscape, head and shoulders, the camera at eye level. Stand where the frame's middle band (where
the title sits) is darker or busier than a white wall: trees, a street, a skyline.

**You:**
> We know models can draw. But can they build in the real world?
> Introducing SAMgo.

**Title:** on the vowel of "Sam", **SAMGO** cuts on in huge flat off-white Futura Medium capitals, ~80% of the frame
width, a little letter-spacing, no shadow (the "QUEENS" title style), settling from 103% to 100% over 8 frames, with
a cinematic boom (free-licence). Hold 2-3 s still, looking into the lens, then cut.
Title frames: `scripts/title_card.py` (Pillow, `Futura.ttc` face 0), overlaid by ffmpeg at the
word's onset from whisper word timing.

---

## 0:09-0:22 · The promise

**Picture:** your hand holds a phone; on its screen, the Camry photo (`runs/video/steps/1_photo.png`). Tap. Lower the
phone. Hard cut to the red LEGO Camry orbiting, doors swinging open, wheels turning (`clips/01_Camry.mp4`).

**VO:**
> This is a photo of a car.
> And this is that car in LEGO. The doors swing. The wheels roll. And every one of its 320 pieces is a real brick
> you can buy and build.

---

## 0:22-0:34 · The question

**Picture:** the montage flickers past, half a second each: 747, helicopter, Cessna, wind turbine. Back to black.

**VO:**
> Turning a picture into a shape is one thing. Turning it into a machine that works is another. So we taught Meta's
> models to build working machinery, from a single photo.

**OST:** one photo → a machine that works

---

## 0:34-1:22 · How it works: every step

Each beat ~8 s. Lower-third names the model in the shot.

**1. SAM 3.1 sees it.** `2_sam31_mask.png`, then `3_sam31_glass.png` / `overlay_windshield.png`.
> SAM 3.1 cuts the car out of the photo, and finds exactly which pixels are glass.

**2. SAM 3D lifts it into 3D.** `4_sam3d_mesh.mp4`.
> SAM 3D turns that one image into a full 3D mesh, windshield marked.

**3. Code makes it LEGO.** `5_voxels.png`, then the resolution sweep page.
> Our code slices the mesh into LEGO-sized cells and lays them out as real plates and bricks.

**4. Muse Spark knows.** `6_spark_layout_side.png`.
> Muse Spark looks at the photo and knows what this machine does: these are wheels, these are doors, and here is
> where they go.

**5. Muse Glimmer builds, and the checker talks back.** Text on screen from `docs/feedback-example.md`: Glimmer's
first answer → the checker's sentence typed out in red → the fix, turning green.
> Muse Glimmer, which we fine-tuned, writes the build. Then a physics checker tests every piece: is it a real part,
> is it held by studs, does it swing clear, does it stand. When something's off, the checker says exactly what, in
> plain words, and Glimmer fixes it.

**OST (typed):** "these hinge parts hold nothing - put each on a part of the side it carries"

**6. It builds itself.** `clips/02_Camry_build.mp4`.
> Then it builds: chassis first, axles sliding through their holes, the body layer by layer, every path proved
> collision-free.

---

## 1:22-1:42 · You can build it

**Picture:** flip through `Camry_manual.pdf`: cover, a parts callout with red arrows, a mid-build step, the parts list.
End on the finished car page.

**VO:**
> And it comes with the instructions. Fifty-six steps, each one checked with exact physics so the model holds
> together at every stage. A parts list ready for BrickLink. Real LEGO, start to finish.

**OST:** 320 parts · 56 certified steps

---

## 1:42-2:12 · It generalizes

**Picture:** `clips/montage.mp4`: 747 fans spinning and flaps moving, helicopter rotors, Cessna propeller and
ailerons, supercar. Close on the 40-stud wind turbine turning.

**VO:**
> Here's what matters most. Glimmer never trained on an airplane, a helicopter or a wind turbine. It learned
> general skills on plain blocks: hinge an edge, set a rotor into a face, put a turntable on top.
> Then it met a 747, and built its engine fans. A helicopter, and built its rotors. A Cessna: propeller and ailerons.
> Same models. Same checker. Nothing written for any one machine.

**OST:** none of these was in the training data

---

## 2:12-2:35 · The numbers

**Picture:** one clean slide, numbers counting up one by one.

**VO:**
> On test sets frozen before training: fifty of fifty cars built. Ten of ten doors on vehicles it had never seen.
> A brand-new skill, upright rotors, from eight to twenty-four of thirty in a single training run.

**OST:**
- 50 / 50 held-out cars
- 10 / 10 doors on unseen vehicles
- 92 / 112 tasks on the first try
- upright rotors 8 → 24 / 30 in one run

---

## 2:35-3:00 · Close (on camera)

**Picture:** you, to camera. Cut to the Camry orbit for the last line, then the end card.

**You:**
> Machines are hard to make. You need to know what every part does and how it moves, and that it will hold
> together when it's built. SAMgo puts that in reach of a single photo. Today it's LEGO. The same idea, a model that
> knows, a model that builds, and code that checks every answer, is how machines will design the machines we build
> next.

**End card:** SAMgo · github.com/vvennela/samgo · Built with SAM 3.1, SAM 3D, Muse Spark, Muse Glimmer · team names ·
photo and footage credits.

---

## Notes for the edit

- Music: one track that builds; the hit lands on the cold-open title and again on the first door swing.
- Sound: a soft brick click on every step in the build animation and the manual flip.
- Keep the montage and numbers on civil machines only (no military objects).
- Record the VO in one sitting, close to the mic, unhurried; re-record any line that rushes.
