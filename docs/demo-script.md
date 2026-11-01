# SAMgo demo video (target 2:30)

Show only checked artifacts: every shot uses a saved, verified model (viewer pages, LDraw files, certified manuals).
Cached runs are labeled "recorded run"; live runs are labeled "live".

| Time | Shot | Voice-over |
|---|---|---|
| 0:00–0:12 | Photo of a VW Beetle, then the finished LEGO model rotating in the viewer | "One photo in. A LEGO model of that exact car out: one that rolls, has glass, and could actually be built." |
| 0:12–0:30 | Pipeline diagram (README mermaid): SAM 3.1 → SAM 3D → builder → Glimmer → verifiers | "Meta's SAM 3.1 cuts the car out, SAM 3D makes the mesh, and our builder turns it into real LEGO parts: symmetric, smoothed, a frame and a hollow shell like a real car." |
| 0:30–0:55 | Terminal: Glimmer's first proposal for the Camry rear axle, the checker's message ("use a longer Technic brick"), the corrected proposal passing | "Then Muse Glimmer makes it work. It edits the model to add Technic running gear. Code checks every edit: structure, stability, motion. When it's wrong, it gets the exact error and fixes it." |
| 0:55–1:20 | Viewer: Beetle with running gear and glass, motion preview turning each axle; underside view | "The axles run through real Technic bricks, the wheels clear the arches, the windows are glass parts LEGO actually makes." |
| 1:20–1:45 | Viewer gallery: chair (casters swivel, seat turns), microwave (door, turntable), car doors, laptop, TV | "The same model places hinges, turntables, casters and crank arms across object types it was trained on: 67 of 72 held-out cases." |
| 1:45–2:05 | Certified manual: step images with "certified" badge; a line from the certificate | "And the instructions are formally certified: every step is checked exactly, no step needs you to hold anything, and an independent checker re-verifies it." |
| 2:05–2:20 | BrickLink Studio opening the LDraw file | "Every part is real: the LDraw export matches the official part library to within half a millimetre." |
| 2:20–2:30 | Results table (README) incl. the limit | "What doesn't work yet: placing a whole mechanism in an object type it has never seen. Repairs do transfer: 20 of 20. That's next." |

## Shot list and sources
- Beetle photo: `photos/beetle.jpg`; mesh: `runs/s6/beetle/object.glb`.
- Live pipeline run (Glimmer, glass, colors, checks): `runs/demo_final/beetle/` (and `runs/demo_r5/` after run 5).
- Feedback correction on the Camry: `runs/demo/camry/r3_seq_log.json` (front round 3, rear round 2).
- Mechanism gallery: `python -m samgo.viewer3d out.html "Chair=runs/demo/chair/chair.json" ...`.
- Certified manuals: `runs/demo/*/<name>_manual.json` (chair, skateboard, microwave; cars if certified).
- LDraw files: `runs/ldraw/*.ldr`.
- Results: README "Results" section; frozen test data under `runs/audit/`.

## Recording notes
- Record the viewer at 1080p with the motion preview on; it animates one checked joint at a time.
- Keep a terminal font large enough to read the checker's message in the feedback shot.
- The voice-over numbers must match the README at recording time (update after run 5).
