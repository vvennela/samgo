# 3-minute demo video (draft)

Show only work from Oct 27 - Nov 4. Civil machines only (no military objects).

**0:00-0:20 Hook.** A photo of a desk fan -> a LEGO fan whose blades spin (3D viewer, motion preview). "One photo of
any machine becomes a LEGO model that works, and that you can actually build."

**0:20-0:50 How: Spark knows, Glimmer does, code verifies.** The diagram from the README.
- SAM 3.1 cuts the object out, SAM 3D makes the mesh, code turns it into real LEGO plates and bricks.
- Muse Spark looks at the photo: "the blades spin, the head tilts, the guard is see-through" and points where.
- Muse Glimmer (our fine-tune) gets the nearby parts and one sentence, and writes the edits that build it.
- Code checks every part is real and held by studs, the mechanism moves clear, the model stands, and the build order
  is certified. Failures go back to Glimmer as exact feedback ("this clip holds nothing").

**0:50-1:30 It generalizes.** Same models, no per-object code:
- Camry: wheels roll, doors swing, windows are glass.
- Wind turbine / airliner: whatever passed (rotor, flaps, landing gear), in the 3D viewer.
- "None of these objects was in the training data. Glimmer learned generic skills on generic blocks: hinge an edge,
  add a leg, set a rotor into a face."

**1:30-2:10 Feedback is the trick.** Screen recording of one case: Glimmer's first answer fails, the checker's
sentence, the fixed answer passes. Then the skills table: rotor 1/30 -> 30/30 after one generic training run; flaps
15 -> 22 within rounds once Glimmer trained on its own mistakes.

**2:10-2:45 You can build it.** The build kit: BrickLink parts list, certified step pictures (49 steps for the Camry),
the model in BrickLink Studio. (If built: the real LEGO model next to the photo.)

**2:45-3:00 Close.** "Meta's models, turning a photo into a working, buildable machine, including machines they were
never trained on." Repo link.

Credits on screen: team members; models: SAM 3.1, SAM 3D, Muse Spark, Muse Glimmer; photo sources from
`docs/photo-sources.txt`.
