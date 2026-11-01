# Finish plan — complete spec by tonight (Sun Nov 1, 23:59 PT)

Goal: every item of the spec (`spec.md` section 0 plus the Oct 30 scope change) done, verified, and packaged for
submission. Started Sun Nov 1, 01:22 on branch `r4-generalize` (contains all of `audit`). Nothing is merged into
`main` without the user's word; pushing stays with the user.

Status at start: core ~80%, full wish list ~65%. Run 4 (Glimmer places/repairs mechanisms) passes 67/72 on held-out
instances of trained families, 20/20 repairs on never-seen families, 0/20 placements on never-seen families, 46/50 cars.

## Definition of done (the spec, item by item)

| # | Spec item | Done when |
|---|---|---|
| 1 | Photo → faithful, buildable LEGO body | Camry and Beetle build from their photos, pass structure/stability; symmetric, smoothed |
| 2a | Glimmer: Technic running gear | photo Camry and Beetle pass live with run 4 (step by step, feedback) |
| 2b | Glimmer: glass windows | windows are trans-clear/trans-black parts on both photo cars, placed by code (Glimmer task only if run 5) |
| 2c | Clean, consistent colors | one body color + accents, mirror-consistent, doors match the body, Technic parts in standard colors, every part/color pair exists in LEGO's catalog |
| 2d | Glimmer: doors | run 4 places hinged doors on both photo cars; doors swing outward 90° clear |
| 3 | Verifiers + feedback | already done (audit-corrected); all tests pass |
| 4 | Demo: photo → car that rolls, has glass, clean colors, opening doors | one command produces it; viewer + instructions + LDraw |
| S1 | Scope: bicycle, chair, skateboard, microwave, laptop lid, TV turntable | each builds, passes its checks, animates; Glimmer places/repairs (run 4); skateboard hinge fixed; bicycle and skateboard at higher resolution |
| S2 | Physically real | every model exports to LDraw with no part-size mismatch over 1 mm; a verified instruction manual (each step stands and can be inserted) |
| S3 | Evidence | frozen run 3 vs run 4 comparison, honest limits (placement on unseen families) |
| P | Submission package | README, write-up, demo video script, viewer links, cached demo artifacts |

## Schedule (each block ends with tests passing and a commit)

| Time | Block | Work |
|---|---|---|
| 01:30–03:00 | A. Colors | `samgo/colors.py`: body palette (main + accent from the photo), mirror and region consistency, standard Technic colors, part/color availability (Rebrickable colors + elements, cached in `runs/rebrickable/`), swap to an existing pair; wire into the builder; tests |
| 03:00–04:30 | B. Glass | window detection on the mesh/voxels (greenhouse band, windscreen/rear slopes, mirrored), recolor to trans-clear/trans-black parts that exist; viewer + LDraw already support them; tests |
| 04:30–06:00 | C. Skateboard + hi-res | click hinges end to end (LDraw mismatch → 0); bicycle ~32 long with largest wheels, skateboard 6×30 with kick tails; every check passes; tests |
| 06:00–07:00 | D. Small-car gap | why `car/gear` synthetic cars pass 1/6 under run 4; fix the data/prompt or the posing |
| 07:00–09:30 | E. Demo pipeline | `pipeline.py` uses run 4: body → colors → glass → Glimmer running gear (step by step) → Glimmer doors → final checks → viewer (with motion preview), instructions, LDraw; run on Camry and Beetle photos (cached meshes and fresh perception); fix what fails |
| 09:30–12:00 | F. Verified manual | `samgo/manual.py`: per step, the partial build stands (structure + stability, mechanisms locked) and each new part can go in (straight down; axles/pins along their axis) without hitting placed parts; gear as sub-assemblies; manual inventory == LDraw inventory; fix the step order where a check fails; run on every demo model |
| 12:00–13:00 | G. Physical check | re-export all models to LDraw; mismatches over 1 mm = 0; user opens one in BrickLink Studio (10 min) |
| 13:00–16:00 | H. Run 5 (background, optional) | placement on unseen families: randomized mechanism layouts on arbitrary bodies + glass task; train free; frozen test on the same r4-frozen-v2 cases; claim only if it improves held-out placement |
| 13:00–17:00 | I. Evidence + README | README rewrite (what Glimmer does / code does / checks establish), results tables, limits, how to run; fix local paths in docs |
| 17:00–19:00 | J. Demo assets | final viewer pages (photo cars with glass, colors, doors, running gear; chair, bicycle, skateboard, microwave, laptop, TV), instructions PDFs, video script and shot list |
| 19:00–21:00 | K. Buffer | failures from E/F/H; full test suite; re-run frozen evaluation if run 5 lands |
| 21:00–23:59 | L. Wrap | final commit; summary for the user; list of user-only steps |

Runs in parallel where independent: H (training) overlaps I–J; A/B/C touch separate files.

## Decisions taken by default (to finish tonight)
- Glass is placed by code; a Glimmer glass task joins run 5 only.
- Work stays on `r4-generalize`; merging into `main` waits for the user.
- If Fireworks stalls, use the last saved checkpoint and cached runs, labeled as such.
- Paid inference: only the live demo runs and (if run 5 trains) one frozen evaluation; usage has been rating at $0.

## User-only steps (cannot be done by code)
1. Say whether to merge `r4-generalize` into `main`.
2. Rotate the Fireworks API key printed in an earlier error, before the repo is public.
3. Open one exported `.ldr` in BrickLink Studio; optionally build one model.
4. Record the demo video from the script; submit by Wed Nov 4, 6 PM PT; post on X.
