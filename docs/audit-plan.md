# SAMgo audit and completion plan

Created October 31, 2026 on branch `audit`.
Starting revision: `1004984b543abe01e7df42edf60dd47113297e12`.

## Objective

Demonstrate that a model can select and connect discrete components to create an assembly with a desired shape and function, using executable feedback to correct its mistakes. LEGO is the initial constrained assembly domain; CAD components are a future extension requiring their own geometry, mating, tolerance, load, and assembly-process validation.

The audit will make the demonstrated capability trustworthy, reproducible, and easy to distinguish from the procedural scaffolding that supports it. It will preserve a focused path to the November 4 submission recorded in `spec.md`, while documenting deeper research gaps explicitly.

The workstreams below describe the implementation plan; the completion record identifies implemented slices. This supplements the existing scope in `spec.md`; it does not silently expand the submission requirements.

## Starting evidence and working rules

- The previous local test run passed 152 tests in 48 seconds with the documented external environment and Matplotlib addition.
- Targeted probes nevertheless found overlapping wheels accepted by the full running-gear check, inconsistent edit interpretation between `explain()` and `apply()`, fractional axle lengths accepted by `_validate()`, and instruction generation incompatible with current wheels and bushes.
- The latest saved frame-task evaluation reports 48/50 first-try successes across 50 vehicle IDs. All 50 reconstructed prompts matched the stored prompts; all 48 previously passing answers passed the current checker on replay. This is historical evidence, not evidence under the corrected checker planned here.
- The current photo entry point uses the procedural car builder. The learned editing demo runs separately; broader mechanism builders are largely procedural.
- `samgo/ldraw.py` was already untracked when this branch was created. Preserve it as existing work; do not change or commit it incidentally.
- Preserve existing datasets, checkpoints, and evaluation logs. Write audit outputs to a separate run directory and identify historical results by their original checker and task versions.
- Work in small commits organized by invariant or user-visible behavior. Each correctness fix gets a regression that fails before the fix and passes afterward. Avoid a broad rewrite or tests that merely repeat implementation details.
- Cloud sampling and training are separate execution steps with recorded configuration and cost. This plan does not launch them or promise that new training is needed.

## Workstream 1 — Establish a reproducible baseline

**Purpose:** separate existing defects from changes introduced during the audit.

1. Inventory the part representations, task variants, entry points, outputs, and checkers used by the current car and mechanism demos.
2. Record revision, environment, dependency versions, fixture hashes, and test command in an audit run manifest. Run the existing suite before changes.
3. Turn the confirmed failures into small deterministic regression cases. Add cases for contradictory edits, invalid indices, unknown operations, unsupported dimensions, and duplicated functional parts.
4. Inventory every part type actually emitted by the demo builders. Record which consumers support it: validation, geometry, connectivity, motion, stability, rendering, instructions, and export.
5. Make core tests independent of ignored local runs. Prefer small synthetic fixtures; give photo integration tests explicit fixture requirements and a separate marker. A missing integration fixture must be reported rather than silently counted as a pass.
6. Declare the dependencies needed by supported local commands and document or pin the external training runtime. Add a minimal CI path for the portable tests.

**Done when:** a clean checkout can run the portable suite with documented setup, and the original defects have reproducible failing cases. Fixture-dependent and network-dependent checks are identified separately.

## Workstream 2 — Make edit execution and part validity authoritative

**Purpose:** ensure the assembly checked is exactly the assembly saved and shown.

1. Introduce one strict edit contract. Validate operation names, part objects, integer indices, index bounds, and each task's permitted operations. Reject negative indices and ambiguous repeated operations on the same original part.
2. Use one application function for training generation, evaluation, feedback, and the demo. Return the candidate assembly plus provenance mapping input/local indices to added, changed, and retained parts.
3. Make feedback describe structured issues on that candidate. Remove independent reconstruction in `explain()` and replace message-substring filtering with issue codes and explicit partial-task policies.
4. Give partial steps precise postconditions: required gear for this axle line exists; new parts are attached; prohibited collisions are absent; previously accepted work is preserved. A partial pass must not imply whole-object stability or function.
5. Validate parts at the boundary before geometry or numerical solvers run. Reject booleans-as-numbers, nonfinite values, invalid dimensions, invalid axes, and nonexistent catalog combinations. Preserve explicitly supported half-step placements.
6. Consolidate common part metadata incrementally: legal dimensions, units, wheel widths, connector definitions, and rendering/export identity. Start with the submission's supported parts; migrate consumers behind compatibility adapters rather than rewriting everything at once.

**Done when:** every accepted edit list produces the exact candidate that was checked; malformed or prohibited edits fail with actionable errors; failed candidates do not replace the last accepted assembly; all submission part types have a documented validation contract.

**Primary code:** `samgo/running_gear_task.py`, `samgo/demo_running_gear.py`, `samgo/eval_running_gear.py`, `samgo/parts.py`, `samgo/technic.py`.

## Workstream 3 — Strengthen physical and functional checks

**Purpose:** close demonstrated false positives and define what each verdict establishes.

1. Add collision checks covering functional parts, including duplicate and intersecting wheels. Use spatial bounds to find candidate pairs, followed by appropriate shape tests. Legitimate axle-in-hole and mating contact must be exempted at the connection geometry, not by ignoring entire parts indiscriminately.
2. Check that a car's wheel arrangements occupy distinct positions on the required sides, attach to the intended axle lines, and provide a consistent ground-contact configuration. Check retention and support using actual engagement spans.
3. Bind motion requirements to the intended component or assembly group. A nearby unrelated joint must not satisfy a door, crank, steering, or wheel-spin requirement. Require wheel spin about the wheel's own axle axis.
4. Audit sweep coverage: initial pose, endpoints, both required directions, moving wheels, joint partner geometry, and collisions between moving assemblies when simultaneous motion is claimed. Make sampling resolution explicit. Add small adversarial cases at tolerances and between angular samples.
5. Separate schema/catalog validity, connectivity, collision clearance, estimated static stability, motion satisfaction, and assembly-order feasibility. Return `pass`, `fail`, or `unknown/not evaluated` with the applicable assumptions.
6. Correct support modeling for the selected demo's actual contact geometry where feasible. Report remaining idealizations, including locked joints and uncalibrated clutch force. Do not claim all-pose stability from a closed-pose LP.
7. Remove the viewer's unchecked 60-degree fallback. Display only checked motion ranges as verified; unknown ranges remain explicitly unknown. Keep visualization and checking on the same motion description.
8. Check preservation of intended appearance separately from adherence to the modified assembly target. Report original-target fidelity and post-construction-target fidelity with their own definitions.

**Validation:** perturb known-good assemblies by removing holders, duplicating parts, shifting an axle, blocking a door, locking a wheel, altering support heights, or adding an unrelated joint. Each perturbation should change the relevant verdict without unrelated checks masking the failure. Include rigid translations and equivalent orientations as invariance checks where supported.

**Done when:** the demonstrated false positives are rejected, valid reference mechanisms still pass, and every displayed success corresponds to a named, bounded check. Continuous collision guarantees, full dynamics, and exhaustive joint configuration search remain future work unless actually implemented.

**Primary code:** `samgo/technic.py`, `samgo/kinematics.py`, `samgo/stability.py`, mechanism-specific check adapters, `samgo/viewer3d.py`.

## Workstream 4 — Connect one reliable end-to-end path

**Purpose:** make the trained assembly capability directly runnable from the product entry point.

1. Create a single orchestrator for photo/mesh input, body construction, task creation, model edits, candidate checking, feedback, final validation, and output generation. Reuse the task implementation rather than copying the demo loop into the pipeline.
2. Make procedural reference construction and learned mechanism placement explicit modes. A procedural fallback must be labeled in the result; it cannot count as model success.
3. Record the spec, supplied regions, model-visible parts, raw answers, parsed edits, candidate reports, accepted states, and final assembly. Save model/checker/task identifiers, token budgets, retry counts, and timings.
4. Key perception and intermediate caches by relevant input hashes and configuration. Rehearsal caches must be visible in the run manifest and must not reuse another input's result silently.
5. Define retry limits, timeouts, parse failures, and partial-output behavior. Only accepted edits advance the build; preserve failed proposals for the demo's explanation.
6. Repair instruction generation for current wheel sizes and bushes. Every final part must appear exactly once in the plan and bill of materials. Unsupported export or instruction types must produce an explicit result rather than being dropped.
7. Finish with one aggregate report. Require the checks relevant to the selected capability before labeling the output successful. Static instability must not be presented as a fully passing build.

**Done when:** one documented command completes a cached photo-car run through the learned editor and exports the viewer, reports, and supported instructions. A fresh-perception run is also exercised separately; recorded timings distinguish cached and fresh paths.

## Workstream 5 — Freeze evaluation and test transfer

**Purpose:** establish what the model learned under a fixed, defensible experiment.

1. Freeze datasets and task instances after the correctness fixes. Store input assemblies, local numbering, requirements, targets, reference answers, split membership, and hashes; do not reconstruct a benchmark differently as code evolves.
2. Split by source object, preserving all scales and variants in the same split. Keep previously inspected photo examples as development demonstrations and reserve untouched objects/configurations for final evaluation.
3. Recheck reference answers and saved model predictions under the corrected verifier. Publish old and new results separately and categorize any changed verdicts. Record rejected bases and generation failures so conditional task success is distinct from end-to-end coverage.
4. Compare base and tuned Glimmer on identical task instances, prompt information, token budgets, sampling settings, and feedback budgets. Report truncation, parse failures, and API failures separately.
5. Report first-try task success, success after feedback, full multi-step assembly success, function-specific outcomes, latency, tokens, and original-target fidelity. Include sample counts and uncertainty; avoid presenting per-axle success as per-car success.
6. Compare feedback with an equal-budget retry-without-feedback condition if resources permit. Keep the procedural reference as a visible baseline and record which decisions it supplies to the model.
7. Stress transfer through changed dimensions, orientation, spacing, and constrained part availability, where reference solutions establish feasibility. Clearly separate transfer to new objects within a mechanism family from transfer to an unseen mechanism family.
8. Consider a second learned mechanism family, preferably a small hinge task using existing infrastructure, only after the main path and checker are stable. Generate varied verified examples, train on final correction turns, and hold out configurations. If unfinished, retain broader mechanisms as labeled procedural demonstrations.

**Done when:** the submission can make a precise claim supported by a frozen experiment, a matched baseline, and examples whose solution decisions are attributable to the model. Additional training is justified by failure analysis, not by the existence of spare infrastructure.

## Workstream 6 — Establish assembly feasibility and physical evidence

**Purpose:** connect a valid digital arrangement to an object a person can assemble and use.

1. Select a compact build using obtainable parts and produce an accurate inventory. Record substitutions; any substitution changes the assembly to be checked.
2. Replace pure height ordering where necessary with explicit subassemblies and precedence constraints: mount retainers and wheels in the correct order, preserve axle insertion access, and attach moving components without trapping their connections.
3. Add targeted insertion/accessibility checks for the selected build. Do not label these as a general assembly planner. Temporary support or manual reorientation can be legitimate, but instructions must state it.
4. Have a person follow the plan and record mismatches: unavailable parts, inaccessible connections, friction, slipping wheels, sagging members, or insufficient clearance. Feed measured failures back into model assumptions and tests.
5. Demonstrate each claimed function directly: rolling, free wheel rotation, door opening, or steering. For mechanisms without a drivetrain or coupling, describe the demonstrated degrees of freedom accurately.

**Done when:** the chosen physical assembly and its observed functions are documented, or the submission explicitly limits its evidence to computational checks. Physical construction is a human/external dependency that code execution cannot complete on its own.

## Workstream 7 — Present the evidence and the CAD extension accurately

1. Update the README and run instructions to describe the implemented pipeline and current results. Label trained, procedural, verified, approximate, and planned capabilities consistently.
2. Structure the demo around intent, model choices, a meaningful rejected proposal, correction, and the resulting function. Show the exact checked assembly rather than an independently prepared visual.
3. Include the matched evaluation, known limitations, and one honest failure example. Prepare cached rehearsal artifacts with provenance and distinguish them from fresh model execution.
4. Explain the CAD path concretely: replace the catalog and connector model with CAD components and mating constraints; add tolerance, load, contact, and process checks. Treat general CAD assembly as a next stage of validation, not an already established result.

**Done when:** a reviewer can reproduce the supported workflow and identify exactly what the model learned, what the algorithms supplied, and what the checks and physical evidence establish.

## Execution order and scope control

| Priority | Work | Dependency / exit gate |
|---|---|---|
| P0 | Baseline and defect regressions | Preserve current evidence; reproduce failures |
| P0 | Canonical edits and strict part validation | Checked candidate equals applied candidate |
| P0 | Functional overlap, wheel placement, and motion verdict fixes | Known false positives rejected; reference builds pass |
| P0 | Integrated learned car path and accurate outputs | One command completes the demonstrated workflow |
| P1 | Corrected benchmark replay and matched model evaluation | Freeze task and checker before new scores |
| P1 | Physical assembly, instructions, and demonstration | Begin inventory/access planning early; record real observations |
| P1 | README, evidence, and submission preparation | Leave dedicated presentation/rehearsal time |
| P2 | Second learned mechanism family | Proceed only after P0 gates and a usable benchmark |
| Later | General assembly planning, all-pose stability, CAD component ingestion | Separate research milestones with independent validation |

The first implementation slice is the edit contract plus its regression cases, followed by wheel overlap/placement and instruction compatibility. These fixes address observed failures without expanding the mechanism scope. Reassess the remaining effort after that slice; the earlier broad completion estimate is not a commitment to finish every research item before submission.

## Completion record

For each implemented slice, record the change, tests executed, evidence artifact paths, and remaining assumptions. Keep a short unresolved-issues list with severity and the claim each issue limits. No workstream is complete merely because its current tests pass; its exit gate must also be satisfied.

### First slice — edit agreement, wheels, and instructions (October 31, 2026)

Implemented on `audit`:

- `samgo/edits.py` is the shared edit executor for the current task and legacy agent. It validates the complete edit list, preserves original local numbering, rejects conflicting edits and prohibited operations, and returns a detached candidate with provenance. Frame tasks enforce add-only edits.
- `explain()` and `check()` now use the same structured checks on that candidate. Feedback formatting and truncation cannot change the verdict. Partial steps explicitly require the requested wheel pair, permit only the existing distant body support deferral, and defer whole-car stability. The demo preserves its input build when every proposal fails.
- Part validation rejects fractional dimensions, strings/booleans used as numbers, and nonfinite coordinates while retaining supported half-step connector placements. Parsing accepts a single JSON edit list (including `[]`), optionally fenced.
- Wheel collisions use cylinder geometry for parallel and perpendicular x/y axes. Wheel counting uses distinct part indices; car wheel pairs must straddle their supporting rows. The chair adapter exempts its independent caster forks from the car placement rule. Catalog wheel size determines the mounting span consistently.
- Instructions accept modern size-based wheels and both bush types, include every supported part exactly once in steps and inventory, and draw/bound gear in either axis. Unsupported types fail before output creation. Generated instructions state that insertion access and temporary support are unchecked.

Validation:

- Baseline: **152 tests passed** before changes. Full suite after the main fixes: **210 passed in 45.77 seconds**. After adding three malformed-input regressions and cleaning imports/helper naming, the affected edit, Technic, instruction, and chair suites passed **80 tests in 8.49 seconds**.
- Full-suite command: `UV_PROJECT_ENVIRONMENT=/Users/vishnuv/.venvs/samgo uv run --no-sync --with matplotlib python -m pytest -q tests/`. The final targeted command selected `tests/test_running_gear_task.py tests/test_technic.py tests/test_instructions.py tests/test_chair.py`. Ruff passed for all modified Python files; `git diff --check` passed.
- Replayed all 50 saved answers from `runs/eval/r3_tuned.json` against reconstructed frame tasks and the corrected checker: **48/50 pass, zero changed verdicts, zero unavailable references**. All 50 reconstructed prompts match the stored prompts. All 48 recorded application-agreement checks pass; two rows have no application-agreement result. This reuses saved answers; it is not a fresh model evaluation or proof of whole-car success for partial tasks.
- Local replay report: `runs/audit/first-fixes/r3_replay.json`. Rendered saved `r3_seq.json` assemblies for Camry (397 parts, 47 steps) and Beetle (376 parts, 46 steps), both passing the function check. Instructions are under `runs/audit/first-fixes/instructions/{camry,beetle}/instructions.md`. Inspected the Camry finished view and Beetle final step visually. Run artifacts remain ignored local files; tests use synthetic fixtures for the new regressions.

Remaining limits relevant to this slice:

- **P0 — Integration:** the learned demo and photo entry point still need the unified orchestration and aggregate reporting described in workstream 4. This slice does not complete that workstream.
- **P1 — Physical claims:** collision coverage is improved for wheel pairs, not every functional-part pair. Wheel width uses the catalog's stud envelope. Retention, actual contact/support geometry, motion requirement binding, and continuous motion sweeps still need the wider audit. Passing these checks is not physical validation.
- **P1 — Buildability:** instruction completeness and rendering are fixed; insertion order, subassemblies, access, and temporary supports remain unchecked. Unknown mechanism types deliberately reject instead of disappearing from instructions.
- **P1 — Reproducibility:** the existing full suite still relies on local photo fixtures, and Matplotlib still needs the explicit runtime addition above. Portable fixture handling, dependency declaration, and frozen benchmark manifests remain workstream 1/5 tasks.

No cloud sampling or training was launched. The preexisting untracked `samgo/ldraw.py` was preserved.

### Second slice — integrated learned photo pipeline (October 31, 2026)

Implemented on `audit`:

- Extracted the model/check/feedback loop into `samgo/assembly.py`; both the photo pipeline and compatibility demo use it. The sampler receives only prompt messages, never procedural reference answers. Failed proposals preserve the last accepted state. Retry counts, response-wait limits, raw answers, candidate assemblies, feedback, and accepted states are recorded.
- Added explicit procedural/learned modes, checkpoint selection, supplied meshes, sequential/joint tasks, and exact-input/message replay. Reference/target identity is checked before replay. If the reference itself cannot pass a partial task, a verified joint task is selected and the reason is recorded.
- Added a bounded, recorded search over three procedural body seeds. Each candidate must pass the full checker before a learned task is posed. This was necessary because `build_arched_car()` can return its final candidate despite failed stability. Model failure and scaffolding failure now have separate evidence.
- Each invocation has a separate artifact directory; the top-level summary identifies the current run. Cache reuse checks photo/prompt/model/preparation/code identity and mesh hashes. Legacy meshes require explicit `--mesh`; fresh perception is separate from replayed answers.
- Final structure, function and stability checks gate success and instructions. Saved build, viewer and instructions all consume the same accepted assembly. Viewer motion is disabled in this path. Original and shaped targets are saved separately, with original-target fidelity reported independently.
- Declared and locked Matplotlib for local pipeline exports. Instruction bounds are set before drawing to avoid autoscaling incomplete geometry; pipeline tests treat runtime warnings as errors. Live inference still uses the external Fireworks training runtime; setup and commands are in `docs/photo-pipeline.md`.

Validation and live evidence:

- **230 tests passed in 46.46 seconds**, including 17 new integration/orchestration regressions. After final trace metadata and renderer-bounds adjustments, **32 affected tests passed with runtime warnings treated as errors**. Targeted tests cover feedback repair, bounded service failures, preserving a completed first step, exact replay and mismatch rejection, body-search exhaustion, partial-task preflight, cache invalidation, static mode, instability and export failures. Ruff and whitespace checks pass.
- **Live Beetle with supplied historical mesh:** tuned Glimmer completed both axle tasks on their first proposals, using 1,034 generated tokens. All final checks passed: **369 parts, 47 instruction steps, 31.7 seconds total**, including 23.3 seconds in learned assembly. Result: `runs/audit/pipeline-live-beetle/summary.json`.
- **Network-free Beetle replay:** passed in 9.5 seconds and produced a byte-for-byte identical `build.json`. Confirmed the viewer embeds precisely that assembly and no motion animation. Visually inspected the finished instruction image. Result: `runs/audit/pipeline-replay-beetle/summary.json`.
- **Live Camry with supplied historical mesh:** body seeds 0 and 1 failed stability; seed 2 passed. The model completed the first axle but twice proposed a rear beam that left a body part unsupported. The run failed, retained the completed first axle, and emitted no instructions. Result: `runs/audit/pipeline-live-camry/summary.json`.
- **Fresh Camry perception:** SAM segmentation/reconstruction completed in 25.5 seconds. Learned assembly failed the target-similarity gate on its first axle after two proposals. A subsequent run reused the hash-matched perception cache and reproduced this model failure. Reference preflight confirms that the partial task is feasible; the model supplied insufficient supporting structure. The implementation does not lower the threshold or substitute reference gear. Results are retained in `runs/audit/pipeline-fresh-camry/artifacts/`, with the cache-reuse run at its top-level summary.
- These are development integration runs, **not a new benchmark**. The live runs generated 4,714 Glimmer output tokens across nine model responses; perception service usage is separate. No training was launched. Local run artifacts are ignored by Git; their summaries include source/dependency identity and input hashes.

Remaining work: improve learned success on changed body/perception outputs; freeze and evaluate the integrated workflow on held-out objects; broaden physical/motion checks; and validate real assembly access/order. A successful cached-mesh demo and a tested fresh-perception path do not establish reliable success on arbitrary fresh photos. The preexisting `samgo/ldraw.py` remains untouched and uncommitted.


### Third slice — object-family and feedback diagnostics (October 31, 2026)

Implemented a frozen, resumable evaluation across LEGO object families, following the user's clarification that “models” means objects. The 20-case r3 run covers cars, train-style chassis, a bed, chairs, bicycles, a skateboard, microwave mechanisms, car doors and TV/laptop pivots. A four-case single-axle control separates beam selection from multi-axle completeness. Every reference passes and every defective input fails before inference. References are never sent as answers to the sampler.

The main suite yields **2 first-try passes and 2 additional feedback repairs**; generic retry fixes none. One first-call service timeout and three token-limited Beetle door responses are identified separately from geometry failures. Supplemental controls yield **1 first-try pass and 1 additional feedback repair**, with failures under translation and increased beam span. These are development repair/completion probes, not whole-object generation or a held-out benchmark. The multi-axle train tasks fail; the valid train references remain procedural.

Added an offline training inventory with source hashes: the r3 frame dataset contains 1,169 feedback rows but no same-center beam-resizing examples. This is a supervision gap consistent with the errors, not proof of their cause. No training was launched. The initially started base-model sweep was stopped after the scope correction; its output-limit/timeouts do not provide a matched comparison.

Validation: **238 tests passed in 47.58 seconds**; eight audit tests passed after final additions. All **49 parsed main proposals and 10 supplemental proposals** reproduce their saved verdicts, error codes and exact candidate assemblies offline. The CLI can prepare 16 portable cases without historical fixtures, explicitly request local fixtures, render labeled galleries and summarize/recheck saved proposals. Full results, limitations, artifacts and the proposed next training experiment are in [object-audit.md](object-audit.md). The preexisting `samgo/ldraw.py` remains untouched and uncommitted.
