# Demo: video plan and the real build

3 minutes, civil machines only (no military objects), only work from the hackathon (Oct 27 - Nov 4).

## The video

| Time | Shot | How to make it |
|---|---|---|
| 0:00-0:08 | **"SAMgo"** in a big bold sans (the "QUEENS" title style) over royalty-free SF / San Jose skyline footage | Pexels or Pixabay stock (free licence; note the clip URL in the credits) |
| 0:08-0:25 | You photograph a car with your phone, lower the phone - **cut** to the LEGO version of that car rolling across your table | Film the photo moment; the LEGO car is the one you build from this repo's kit |
| 0:25-2:20 | **Every step, visualized**, your voice over each | See "Step visuals" below |
| 2:20-2:50 | You on camera: why this matters, why you built it, how it pushes the frontier | |
| 2:50-3:00 | **Montage of everything else built from a single photo**, each moving | The showcase page: 747 (engine fans), Cessna (propeller, ailerons, flap), helicopter (tail rotor, skid), fan (blades) |

### Step visuals (0:25-2:20), in order

1. **The photo.**
2. **SAM 3.1 finds the object** (its mask) and, from Spark's words ("windshield", "side windows"), **the glass**: `runs/glass_proto/overlay_car_windows.png`, `overlay_windshield.png`.
3. **SAM 3D: the 3D mesh** (the glass marked on it).
4. **Voxels** - the resolution sweep, 1x to 10x: https://claude.ai/artifact/MCUrJq8XbVmcDtatBsuZPy
5. **Spark names the working parts** on the region map (wheels, doors; propellers, flaps on other objects), says which are mirror twins, and reviews the build against the real object (missing parts added, colors, glass).
6. **Glimmer builds them**, the checker answers in plain sentences, Glimmer fixes it: `docs/feedback-example.md` (a real case).
7. **The video-game build**: open the model in the viewer, press **Build it**. The rolling chassis comes first (Technic bricks, axles sliding through their holes, wheels sliding on), then the body bottom-up; each piece arrives along a path proved collision-free in Lean (`formal/AnimationPath.lean`), and it speeds up until the last pieces snap into one model. Screen-record it.
8. **It works**: press **Preview motion** - wheels roll, doors swing.

Viewer pages: showcase https://claude.ai/artifact/WDqu8phpQUANxebg8BWncL, red Camry https://claude.ai/artifact/GDPzA6XN37pSiKQ4Ya4ssa.
To make a page for any build: `PYTHONPATH=. python -m samgo.viewer3d out.html "Name=path/to/build.json"`.

## The real build

**The kit to buy and build: `runs/build_kit/camry_red_final/`** - the red Camry with the photo's glass, two
full-height hinged doors (windows in the doors) and four wheels: **395 LEGO elements, 68 steps, certified** (every
step held by stud forces, checked exactly). **`SHOPPING.md` is the store list** (what the Pick & Build wall has, by
colour, and what to order online); `BUILD.md` has a picture per step; `parts.csv` and `bricklink_wanted.xml` list every
part. (The body is the pipeline's; the doors were cut by hand with the project's door builder, so this kit is
"hand-finished".)

### What to buy
From `runs/build_kit/camry_red_final/` (the hand-finished red Camry: photo glass, two hinged doors, 4 wheels; 395 elements, 68 certified steps). Buy ~10% spares of the small plates.

#### 1. LEGO Store Pick & Build wall (plates and bricks)

Colour matters only for looks: **red** is the body; **black** the chassis; **dark bluish gray** is hidden inside (floor pan, hinge holders) - any colour works for those.

| Part | Red | Black | Dark Bluish Gray |
|---|---|---|---|
| Brick 1x2 (3004) | 1 | - | - |
| Plate 2x4 (3020) | 11 | 6 | - |
| Plate 2x3 (3021) | 11 | 5 | - |
| Plate 2x2 (3022) | 15 | - | - |
| Plate 1x2 (3023) | 26 | 8 | 5 |
| Plate 1x1 (3024) | 48 | 9 | 5 |
| Plate 2x8 (3034) | - | 9 | - |
| Plate 1x8 (3460) | 2 | 3 | 20 |
| Plate 1x3 (3623) | 31 | 13 | 4 |
| Plate 1x6 (3666) | 9 | 4 | 5 |
| Plate 1x4 (3710) | 38 | 13 | 17 |
| Plate 2x6 (3795) | 3 | 4 | - |
| **total** | **195** | **74** | **56** |

#### 2. Not on the wall: online (BrickLink / LEGO Pick a Brick) or from a small Technic/City set

Upload `bricklink_wanted.xml` on bricklink.com (Want -> Upload) to buy exactly these.

| Part | Colour | Count | What it is for |
|---|---|---|---|
| Plate 1x2 (3023) | Trans-Clear | 12 | windows |
| Plate 1x1 (3024) | Trans-Clear | 12 | windows |
| Plate 1x3 (3623) | Trans-Clear | 8 | windows |
| Plate 1x6 (3666) | Trans-Clear | 1 | windows |
| Plate 1x4 (3710) | Trans-Clear | 5 | windows |
| Technic Axle 11 (23948) | Light Bluish Gray | 2 | axle |
| Hinge Plate 1x4 Swivel Base (2429) | Dark Bluish Gray | 4 | door hinge (base) |
| Hinge Plate 1x4 Swivel Top (2430) | Dark Bluish Gray | 4 | door hinge (top) |
| Wheel rim 24x14 with axle hole (2994) | Light Bluish Gray | 4 | wheel rim |
| Technic Brick 1x4 with holes (3701) | Black | 14 | chassis: the axles pass through these |
| Tyre 30.4x14 (6578) | Black | 4 | tyre |

Trans-clear plates: the store wall sometimes has them; trans-black or trans-light-blue also work for the windows.
If you only get the plates today, the car body can be built now and the wheels, axles and hinges added when they arrive.

Build: `BUILD.md` in the same folder has a picture for every step (the first steps are built upside down, then turned over).

### Sources

- LEGO Store Westfield Valley Fair (2855 Stevens Creek Blvd, Santa Clara; Pick & Build wall):
  https://www.lego.com/en-gb/stores/store/westfield-valley-fair
- Store wall contents change over time: https://en.brickimedia.org/wiki/The_LEGO_Store_Valley_Fair_Santa_Clara,_CA,_USA
- Pick a Brick: how cups and online ordering work: https://brickfact.com/blog/guidebooks/lego-pick-a-brick-everything-you-need-to-know
- Small Technic sets: https://brickset.com/sets/list-81208
