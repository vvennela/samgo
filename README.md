# SAMgo

Photo of an object -> a LEGO model of *that* object that actually works: a car that rolls, a door that swings.
Built for the Meta Global AI Developer Hackathon (submission Wed Nov 4, 2026).

| Camry (from a photo) | Beetle (from a photo) |
|---|---|
| ![Camry](docs/images/camry.png) | ![Beetle](docs/images/beetle.png) |

| Training cars (Objaverse models through the same builder) | | |
|---|---|---|
| ![](docs/images/training_car_1.png) | ![](docs/images/training_car_2.png) | ![](docs/images/training_car_3.png) |

Every brick is a real LEGO part (LDraw IDs); wheels sit in the body's own tire outlines on real-length Technic
axles; shape, colors and brickwork are mirror-symmetric.

## How it works

```mermaid
flowchart LR
    photo[Photo] --> sam[SAM 3.1<br/>cut out the object]
    sam --> sam3d[SAM 3D<br/>3D mesh]
    photo --> spark[Muse Spark<br/>what must it do?<br/>car: 4 wheels, 2 axles]
    sam3d --> vox[Voxelize<br/>mirror plane, smoothing,<br/>clean colors]
    vox --> body[Algorithmic builder<br/>real bricks and plates,<br/>mirrored layout]
    body --> glimmer[Muse Glimmer<br/>adds the mechanism<br/>by editing parts]
    spark --> glimmer
    glimmer --> check{Code verifiers}
    check -- errors --> glimmer
    check -- pass --> out[3D viewer,<br/>instructions,<br/>LDraw]
    subgraph check_detail [Verifiers]
        s3a[Structure:<br/>real parts, studs,<br/>no overlaps]
        s3b[Stability:<br/>force and torque<br/>balance per part]
        kin[Motion:<br/>joints from part connectors,<br/>sweep for collisions]
    end
    check -.- check_detail
```

- **Body (code):** the SAM 3D mesh becomes a faithful, buildable LEGO body: mirror-symmetric, smoothed, built as a
  frame and a hollow shell like real cars (`samgo/voxelize.py`, `samgo/builder.py`, cars: `samgo/arches.py`).
  Windows become glass (`samgo/glass.py`); every part gets a color LEGO actually produced (`samgo/colors.py`).
- **Mechanisms (Glimmer):** Glimmer edits the body to make it work: Technic running gear (one axle per step), and
  hinges, turntables, axles, wheels and crank arms on chairs, bicycles, car doors, TVs and laptops. It sees the
  parts near the mechanism and what the object must do; it never sees the reference answer.
- **Verifiers (code):** structure, static force balance, and a generic motion check that finds joints from how
  part connectors meet and sweeps moving parts for collisions (`samgo/verifier.py`, `samgo/stability.py`,
  `samgo/kinematics.py`). Errors go back to Glimmer, which corrects its answer (up to 3 rounds).
- **Outputs:** 3D viewer with a sampled motion preview (`samgo/viewer3d.py`), LDraw files checked part by part
  against the official LDraw library (`samgo/ldraw.py`), and building instructions whose steps carry an exact,
  independently checked certificate (`samgo/manual.py`).

## Results (Nov 1)

What Glimmer learned, measured on frozen test sets made before training (Muse Glimmer 30B, LoRA SFT on Fireworks):

| Test | Run 3 (cars) | Run 4 (all mechanisms) | Run 5 (+ doors on varied cars) |
|---|---|---|---|
| Held-out instances of the 7 trained object families (place or repair the mechanism) | 1/72 | **67/72** | 60/72 |
| Door hinges on cars from **vehicles never seen in training** | - | 0/10 | **7/10** |
| Never-trained families (skateboard, microwave): repair a broken mechanism | 4/20 | **20/20** | **20/20** |
| Never-trained families: place the whole mechanism | 0/20 | 0/20 | 0/20 |
| Held-out cars: Technic running gear (first try / with feedback) | 48/50 | 46/50 | **47/50 / 49/50** |

- **Live from photos** (cached SAM 3D meshes), run 5: the Camry and the Beetle each get Glimmer's running gear and
  Glimmer's door hinges (first try), glass and produced colors; structure, function, stability and the doors' swing
  all pass. With `--buildable` (structure-first layout) the **Camry passes the complete spec end to end with a
  formally certified 49-step manual**; on that Beetle body Glimmer's running gear passes but its door hinges overlap
  (one of the ~30% of door cases run 5 still misses).
- Glimmer learns the mechanisms it is shown and transfers *repairs* to unseen object families, and door placement
  to unseen cars once its data varies; it does not yet *place* a whole mechanism in an object family it never saw.
  Run 5 traded some accuracy on two fixed door cars (where it now places only hinges) for that generality.
- **Physical evidence:** every model exports to LDraw with no part off by more than 0.4 mm from the official library
  geometry. Building instructions are **formally certified** (exact per-step equilibrium with a declared stud-grip
  capacity, insertion paths, no temporary support, steps of at most 6 parts, sub-assemblies, independent checker)
  for the chair, skateboard, microwave and both photo cars (Beetle 42 steps, Camry 41-49 steps, built with
  `--buildable`: bricks laid out for structure first, then colored by the photo; color accuracy 0.82-0.85).

| Built by | What |
|---|---|
| Glimmer (trained) | Technic running gear; door hinges on cars; placing/repairing hinges, turntables, axles, wheels, crank arms in trained families |
| Code (verified) | bodies from photos, glass, produced colors, door openings; chair, bicycle, skateboard, microwave, TV, laptop builders |
| Checks | structure, stability, motion (sampled sweeps); exact certificates for instruction steps under a stated model |

## Run it

Use the external venv (the repo's `.venv` can hang on iCloud):

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/samgo
uv sync --locked --inexact

# Photo -> verified build + instructions (needs SAM/Spark keys in .env)
PYTHONPATH=. uv run --no-sync python -m samgo.pipeline photos/camry.jpg --prompt car --car -o runs/photo-camry
# ... with Glimmer placing the running gear and door hinges (Fireworks sampler checkpoint; see docs/photo-pipeline.md)
PYTHONPATH=. uv run --no-sync python -m samgo.pipeline photos/camry.jpg --prompt car --car --assembly learned \
    --model <sampler path> --doors --buildable --manual -o runs/photo-camry

# Certified instructions and LDraw for any model
PYTHONPATH=. uv run --no-sync python -m samgo.manual runs/demo/skateboard/skateboard.json
PYTHONPATH=. uv run --no-sync python -m samgo.ldraw runs/demo/skateboard/skateboard.json

# Training bases: Objaverse vehicles -> verified LEGO cars at 16/20/24 studs
PYTHONPATH=. uv run --no-sync python -m samgo.gen_bases --workers 8

# Running-gear SFT rows from the bases (train/val split by vehicle)
PYTHONPATH=. uv run --no-sync python -m samgo.gen_running_gear --bases runs/bases --out data/running_gear

# Interactive 3D viewer for any builds
PYTHONPATH=. uv run --no-sync python -m samgo.viewer3d out.html "Camry=runs/s6/camry/arched_build.json"

# Tests
PYTHONPATH=. uv run --no-sync python -m pytest -q tests/
```

## Layout

- `samgo/`: perception, voxelizer, builders, verifiers, mechanism tasks, data generators, viewer.
- `brickgpt/`: the BrickGPT replication on Glimmer (data, decoder, Fireworks training and evaluation).
- `spec.md`: the plan (section 0 is current). `docs/archive/`: earlier plans. `reports/`: research.
