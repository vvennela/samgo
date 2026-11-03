# Exact assembly certificates: scope and use

Implemented October 31, 2026 on `audit`. This is an exact, independently checked certificate for an explicit static assembly model. It is **not an unconditional guarantee of physical buildability**, and the Python checker itself has not been proved sound in a theorem prover.

## What a certificate establishes

For the exact input parts and a caller-selected policy, the checker establishes:

1. Every part is a supported axis-aligned brick or plate with integer stud/plate coordinates, a legal palette footprint and no unmodeled fields.
2. The solid envelopes do not overlap, the final model is one stud-connected assembly, and it has contact with the flat ground plane.
3. The proposed order contains each original part exactly once. Each part can descend vertically from above the whole assembly to its final position without its envelope passing through a previously placed envelope. This is a continuous swept-volume condition, not angular or time sampling.
4. At **every resting prefix**, including the final model, exact rational contact forces satisfy vertical force balance and both horizontal-axis torque balances for every piece. Contacts can only push, not pull. The sum of forces at each shared stud is at most the declared capacity.

The final statement is existence of a gravity equilibrium under these assumptions, not robustness to disturbances or all possible loads. Temporary supports are absent from this model. A human/gripper carries the new part along its placement trajectory; we do not solve the manipulation forces during that motion.

For each placed part `i`, the equations are:

```text
sum(signed incident contact forces) = w_i * l_i * height_i
sum(signed force * (contact_x - center_x)) = 0
sum(signed force * (contact_y - center_y)) = 0
force >= 0
sum(forces at one shared stud) <= declared compression capacity
```

All checks use integer or rational arithmetic. Contact positions are regenerated from the assembly, so a certificate cannot invent ground supports, connector locations or arbitrary forces. The declared policy and exact ordered input JSON are hash-bound to the certificate. The checker takes the policy from its caller, not from the certificate; a certificate for a weaker policy cannot silently authorize a stronger one.

## Implementation and trust boundary

- `samgo/certify.py` asks Z3's exact linear-real-arithmetic solver for a force witness for each prefix. It starts with a deterministic bottom-up order, or accepts an explicit order to check. It does not exhaustively search orders or plan subassemblies.
- `samgo/synthesize.py` uses failed-prefix feedback to propose permanent support columns. It preserves all input parts and positions, searches within explicit part/attempt/time budgets, and independently replays the complete certificate before accepting a candidate. This producer is procedural; it is not a learned model or a proof that the chosen repair is minimal.
- `samgo/certificates.py` checks the witness independently using Python `Fraction`. It does not import Z3 or trust `sat`, solver tolerances, floating-point feasibility, or a saved `passed` flag. Geometry and paths are checked directly with exact interval comparisons.
- `samgo/instructions.py` rechecks a supplied certificate before exporting and preserves the certified one-part-per-step order. It does not regroup certified steps by height. A mismatched/tampered certificate rejects before output creation.
- `samgo/pipeline.py` records a separate certificate report and `certified_under_model` field for each completed build. `--require-certificate` makes missing/unknown/unsupported certification fail the run and suppress success instructions. The original target/function/stability checks must also pass.

This is formal constraint checking with a replayable witness, not a Lean/Coq proof of the implementation. The trusted code includes parsing, catalog definitions, contact generation, the rational checker and its runtime. A theorem-prover proof of checker soundness would strengthen this boundary. Z3 can be removed entirely when checking an existing certificate. Its documentation describes the [arithmetic theories used by the witness producer](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/).

## Assumptions that matter physically

The model uses nominal rectangular part envelopes and assumes compatible stud mating at a shared grid footprint. It does not model stud/tube deformation or interference during engagement. Four contact points lie at offsets 1/4 and 3/4 stud in each shared stud cell. Ground contacts use four inset footprint corners. The ground is flat and rigid.

Each part's weight is its occupied plate-cell count, at its geometric center. The default compression capacity is **250 abstract force units per shared stud**. This is a declared parameter, not a measured strength rating and not the existing tensile-clutch estimate. There are no lateral forces, joint tension, acceleration, fatigue, tolerance distributions or external payloads. Changing the capacity requires generating and checking against the same explicit policy.

The bed fixture carries no occupant/payload in this model. Claims about bearing an external load require an extended load specification, material/contact measurements and appropriate validation. A certificate cannot establish that the object fulfills a user's semantic goal; target matching and function requirements remain separate gates. In particular, a single certified brick must not earn a successful RL reward for a failed car task.

Wheels, axles, Technic bricks, hinges, turntables, custom fields and unknown geometry are **unsupported**, not filtered out or treated as ordinary bricks. Certification currently caps assemblies at 200 parts and coordinates at an absolute value of one million for resource control. Reaching those limits does not establish invalidity.

## Verdicts and incomplete results

`certified_under_model` means all checks and every prefix witness passed. `invalid_geometry` identifies a violation of this model's discrete geometry/connectivity conditions. `unsupported` means the input exceeds the specification or resource scope. `unknown` means the producer found no witness in its selected order/budget, including compression-only infeasibility for a prefix. `not_certified` can identify an invalid/blocked requested order. **None of these failures proves that no physical construction exists.** A different order, subassembly, tensile stud connection or temporary jig may work.

The timeout is a budget shared across prefix solving; parsing, constraint construction and independent replay also take time. Z3's timeout is not a hard process-level wall-clock deadline. No partial witness is accepted as a complete certificate.

Existing non-strict pipeline runs can still pass their historical heuristic checks while `certified_under_model` is false. Their viewer explicitly says buildability is not certified; their schematic instructions retain the unverified-access warning. Use the strict flag when a certificate is required.

## Commands

Install the locked project dependencies, including `z3-solver`, using `uv sync`.

```bash
uv run python -m samgo.certify prove tests/fixtures/certifiable_bed.json \
  -o runs/audit/formal-v1/bed-certificate.json \
  --instructions runs/audit/formal-v1/bed-instructions

uv run python -m samgo.certify check tests/fixtures/certifiable_bed.json \
  runs/audit/formal-v1/bed-certificate.json

uv run python -m samgo.pipeline photos/example.jpg --prompt bed --static \
  --require-certificate -o runs/certified-bed
```

The first two commands are local and network-free after dependency installation. The photo command uses the existing perception services unless given a supplied mesh. `--compression-per-stud 100` sets a different declared capacity on both `prove` and `check`. `--order order.json` supplies an explicit list of original part indices to `prove`. CLI exits are zero only on successful certification/replay; unsupported and unknown results are nonzero and have no success certificate.

## Validation and current object coverage

A checked-in 12-part bed fixture produces a certificate for all 12 prefixes, which the independent checker accepts. Its finished instructions were visually inspected. The fixture is a procedural example, not newly demonstrated learned assembly. A supported three-part bridge and translated/quarter-turned versions certify as well.

Adversarial regressions reject modified builds, altered forces, negative/float witnesses, missing stages, duplicate order indices, forged specifications and changed caller capacities. Tests cover overlap, disconnected parts, below-support floating models, blocked insertion paths and an assembly with feasible final equilibrium but an infeasible intermediate prefix. Resource exhaustion is reported as unknown, never success. Pipeline tests exercise strict acceptance and rejection and certificate-bound instruction export.

The 20 reference assemblies from the previous object audit were checked without inference. **19 are unsupported because they contain mechanisms; the original 26-part bed is unknown because its bottom-up order has an unstable prefix.** These are verification coverage results, not additional learned-model failures. Artifacts are in `runs/audit/formal-v1/`; the existing audit fixtures were not changed to make them pass.

## Producing a certified repair

The next slice turns verification into feedback for construction. For supported inputs whose geometry already passes, `samgo.synthesize` proposes ground-to-piece columns made of catalog 1x1 bricks and plates under the first uncertified piece. Candidate ordering spreads the support footprint; a depth-first search tries additions within its limits. Every candidate goes through the same geometry, insertion, equilibrium and independent certificate checks. The acceptance specification is unchanged.

```bash
uv run python -m samgo.synthesize build.json -o runs/support-repair \
  --max-added 24 --max-attempts 128 --timeout 30
uv run python -m samgo.certify check runs/support-repair/build.json \
  runs/support-repair/search.json
```

Use a new output directory. `search.json` records the original hash, specification, budgets, attempted additions and failure feedback. Only successful searches export `build.json` and certificate-bound instructions. Original parts remain in their original list positions; `added_indices` identifies the permanent additions. Unsupported parts are never stripped. A timeout or exhausted search has no accepted build or certificate. The time budget has the same non-hard-deadline limitations as the witness producer.

The original **26-part audit bed** now has a **50-part certified repair**: 24 added support bricks, 13 candidate evaluations, approximately six seconds of search on the development machine. Every original part/position is preserved and all 50 resting prefixes certify. The earlier center-first proposal heuristic exhausted its 30-second budget and remains recorded as unknown. Both runs are retained under `runs/audit/formal-v2/`; the final CLI output with replayable candidate additions is `certified-repair/`. These are development examples used to improve the proposal heuristic, not held-out transfer evidence. The completed repair and its instruction rendering were inspected.

The search does not optimize part count, repair overlaps/disconnected inputs, move existing parts, search placement orders, plan temporary supports, or handle mechanisms. Added columns may change appearance or obstruct a desired use; target fidelity and function still need their own acceptance checks. It is deliberately an explicit command, not a silent photo-pipeline fallback. The learned editor described below now supplies proposals through the same certificate gate; the procedural search result alone is not evidence of learned success.

## Learned proposals through the certificate gate

The next implementation adds `samgo.certificate_repair`: the existing r3 checkpoint supplies add-only proposals, receives certificate/task failures, and retries within fixed limits. It preserves every original part and checks allowed support bounds before certification. The separate `samgo.eval_certificate_repair` runner freezes feasible development cases and compares feedback with generic retries starting from the same failed answer. No weights change, and a failed learned proposal never receives a procedural fallback. See [certificate-repair.md](certificate-repair.md) for the protocol, commands and measured results. Mechanism certificate coverage remains unchanged.

To extend the guarantee to cars and trains, the next work is explicit axle/wheel/retainer solids and mating geometry, axial insertion/subassembly plans, wheel-to-ground load transfer, then continuous swept-volume guarantees for required motions. Real part dimensions, tolerances, insertion/contact strengths and declared external loads must accompany that extension. Adding the words “formal” to the existing sampled mechanism checks would not supply those guarantees.

## Merge rule (Lean 4)

`formal/MergeRule.lean` proves the part-merging rule of `samgo/consolidate.py` for every position and size
(`lean formal/MergeRule.lean`, Lean 4 core, `omega`): side-by-side merges along x and y and three stacked plates
into a brick fill exactly the cells of the pieces they replace, which never overlapped; every top stud and bottom
tube an outside part could grip is still there; the only faces lost are internal ones the replaced pieces gripped
each other with. Checked at run time instead: the merged size exists in that color, one color, one rigid body.
`tests/test_consolidate.py` checks that the Python builds exactly the proved pieces (20,000 random pairs) and keeps
every cell and color.
