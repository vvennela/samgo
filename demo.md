# Demo: video plan

3 minutes, civil machines, all work from the hackathon (Oct 27 - Nov 4). The full cut, assets and numbers are in
`video_handoff.md`; the word-for-word script is `docs/video-script.md`.

## The video

| Time | Shot | How to make it |
|---|---|---|
| 0:00-0:10 | **"SAMgo"** in a huge bold sans (the "QUEENS" title style) over SF / San Jose skyline footage | Pexels / Pixabay stock (clip URL in the credits) |
| 0:10-0:20 | Your hand holds a phone showing the Camry photo; lower the phone -> cut to the LEGO car, doors swinging, wheels turning | `runs/video/steps/1_photo.png`, `runs/video/clips/01_Camry.mp4` |
| 0:20-1:20 | **Every step, visualized**, your voice over each | "Step visuals" below |
| 1:20-1:45 | **You can build it**: flip through the manual | `runs/build_kit/camry_red_current/Camry_manual.pdf` |
| 1:45-2:15 | **It generalizes**: 747, helicopter, Cessna, supercar, each moving | `runs/video/clips/montage.mp4` |
| 2:15-2:40 | Numbers | `video_handoff.md` section 5 |
| 2:40-3:00 | You on camera: why it matters, credits | |

### Step visuals (0:20-1:20), in order

1. **The photo.**
2. **SAM 3.1 finds the object** (its mask) and, from Spark's words ("windshield", "side windows"), **the glass**:
   `runs/video/steps/2_sam31_mask.png`, `3_sam31_glass.png`, `overlay_windshield.png`, `overlay_car_windows.png`.
3. **SAM 3D: the 3D mesh** with the glass marked: `runs/video/steps/4_sam3d_mesh.mp4`.
4. **Voxels**: `runs/video/steps/5_voxels.png`; the resolution sweep, 1x to 10x:
   https://claude.ai/artifact/MCUrJq8XbVmcDtatBsuZPy
5. **Spark names the working parts** and points where they go (`runs/video/steps/6_spark_layout_side.png`), says
   which are mirror twins, and reviews the build against the real object.
6. **Glimmer builds them**, the checker answers in plain sentences, Glimmer fixes it: `docs/feedback-example.md`.
7. **The video-game build**: the rolling chassis first (Technic bricks, axles sliding through their holes, wheels),
   then the body bottom-up; each piece arrives along a path proved collision-free in Lean
   (`formal/AnimationPath.lean`): `runs/video/clips/02_Camry_build.mp4`.
8. **It works**: wheels roll, doors swing.

Viewer pages: showcase https://claude.ai/artifact/WDqu8phpQUANxebg8BWncL, red Camry
https://claude.ai/artifact/GDPzA6XN37pSiKQ4Ya4ssa. A page for any build:
`PYTHONPATH=. python -m samgo.viewer3d out.html "Name=path/to/build.json"`.

## The build kit

`runs/build_kit/camry_red_current/`: 320 LEGO elements, 56 certified steps (every step held by stud forces, checked
exactly), `Camry_manual.pdf` (60 pages, LEGO style), `parts.csv`, `bricklink_wanted.xml` (upload on bricklink.com:
Want -> Upload) and `model.ldr` for BrickLink Studio.
