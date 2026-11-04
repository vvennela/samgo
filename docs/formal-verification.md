# Verification: certified build steps and Lean proofs

SAMgo checks buildability with exact arithmetic and proves its geometric rules in Lean 4.

- **Every build step is certified** by `samgo/manual.py` with exact rational force witnesses.
- **The merge rule** (`formal/MergeRule.lean`) and **the build animation's paths** (`formal/AnimationPath.lean`) are
  proved in Lean 4 for every position and size.

## Certified build steps (`samgo/manual.py`)

The manual builds the body upright on the table, one layer per step, then adds the running gear and other axle-line
mechanisms as sub-assemblies. For every step, in exact rational arithmetic:

1. **Inventory**: the manual's parts are exactly the model's parts.
2. **Insertion**: each body part goes straight down without crossing a placed part (a continuous swept-volume
   condition); each axle slides in along its line through Technic holes only; each wheel, bush or half bush slides
   on from its own side without meeting a placed part.
3. **Equilibrium**: after each step every placed piece is in exact force and torque balance: unit weight per plate
   cell, four contact points per shared stud, the table pushing up only, stud grip at most 250 units per stud (the
   same rule as `samgo/stability.py`).
4. **Connected**: the finished body is one stud-connected assembly.

Z3's exact linear-real-arithmetic solver produces the force witnesses; `check` re-derives every contact and re-checks
every witness with Python `Fraction`, without Z3. The red Camry certifies in 56 steps
(`runs/build_kit/camry_red_current/`).

```bash
uv run python -m samgo.manual runs/demo/chair/chair.json   # writes <stem>_manual.json beside it
```

## Static assembly certificates (`samgo/certify.py`)

For brick-and-plate assemblies, `samgo/certify.py` produces a certificate for every resting prefix of a build order;
`samgo/certificates.py` checks it independently (exact interval geometry, `Fraction` forces, no trust in a solver's
`sat` or a saved flag). The policy and the exact ordered input are hash-bound to the certificate, and the checker
takes the policy from its caller. `samgo/instructions.py` re-checks a certificate before exporting instructions.
`samgo/synthesize.py` adds permanent support columns where a prefix needs them and re-certifies the whole build: the
26-part audit bed becomes a 50-part certified build in about six seconds.

For each placed part `i`:

```text
sum(signed incident contact forces) = w_i * l_i * height_i
sum(signed force * (contact_x - center_x)) = 0
sum(signed force * (contact_y - center_y)) = 0
force >= 0
sum(forces at one shared stud) <= declared compression capacity
```

```bash
uv run python -m samgo.certify prove tests/fixtures/certifiable_bed.json \
  -o runs/audit/formal-v1/bed-certificate.json --instructions runs/audit/formal-v1/bed-instructions
uv run python -m samgo.certify check tests/fixtures/certifiable_bed.json runs/audit/formal-v1/bed-certificate.json
uv run python -m samgo.synthesize build.json -o runs/support-repair --max-added 24 --max-attempts 128 --timeout 30
```

Adversarial tests reject modified builds, altered forces, negative or float witnesses, missing stages, duplicate
order indices, forged specifications and changed capacities.

## Merge rule (Lean 4)

`formal/MergeRule.lean` proves the part-merging rule of `samgo/consolidate.py` for every position and size
(`lean formal/MergeRule.lean`, Lean 4 core, `omega`): side-by-side merges along x and y and three stacked plates
into a brick fill exactly the cells of the pieces they replace, which never overlapped; every top stud and bottom
tube an outside part could grip is still there; the only faces lost are internal ones the replaced pieces gripped
each other with. Checked at run time: the merged size exists in that color, one color, one rigid body.
`tests/test_consolidate.py` checks that the Python builds exactly the proved pieces (20,000 random pairs) and keeps
every cell and color.

## Animation paths (Lean 4)

`formal/AnimationPath.lean` proves that the build animation (`samgo/animation.py`) never draws one part through
another: a part sliding in along one axis never meets a placed part whose projections on the other two axes are
disjoint or which lies on the far side, and an axle slid along its line stays in the channel that a Technic brick's
hole leaves free. The Python checks exactly these hypotheses for every move.
