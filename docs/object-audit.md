# LEGO object and feedback audit

October 31, 2026; branch `audit`, following `dd8b81f`.

The scope is LEGO objects, not a sweep of AI checkpoints. The current r3 tuned Glimmer was tested on cars, chairs, a bed, bicycles, a skateboard, microwave mechanisms, car doors, a TV pivot, a laptop hinge, and four-/six-axle train-style chassis. These are development diagnostics. They do not establish general object generation or unseen-photo success.

## What was tested

Twenty frozen cases cover three task types: adding running gear to a supplied body, adding supports to a supplied bed, and repairing known defects in otherwise complete mechanism assemblies. Every case's reference edit passes its checker and every unedited defective input fails. Reference answers are excluded from model messages. Sixteen cases are generated without historical run fixtures; four use saved car-door builds and the previous two failed Camry pipeline steps.

Each first proposal uses temperature 0 and a 4,000-token cap. For each failed response, two independent continuations start from that exact same response: one receives verifier diagnostics and the other receives only a generic request to retry. Both receive one additional response, temperature 0.7, the same token cap, and a 60-second response wait. Call order is counterbalanced by case ID. A timeout or malformed output is recorded separately from a geometric rejection. No training was launched.

This is one trial per case, with nonzero-temperature corrections and no confidence intervals. Feedback recovery is suggestive behavior on these inputs, not a statistically established treatment effect. We inspect observable edits and errors; this does not reveal the model's hidden reasoning.

The four supplemental single-axle cases leave the other axle assembled. They isolate beam selection from the requirement to complete several axle lines. Their generated fixture hash reproduces the evaluated bundle exactly.

## Measured results

| Object/task family | Cases | First pass | Additional passes with feedback | Still unresolved |
|---|---:|---:|---:|---|
| Synthetic cars, two axles | 4 | 0 | 1 | 6-/8-stud span and translated geometry |
| Train-style chassis | 2 | 0 | 0 | Missing axle lines, short beams |
| Bed supports | 1 | 0 | 0 | Missing feet, out-of-bounds additions |
| Chair repairs | 2 | 1 | 1 | Both repaired within the feedback budget |
| Bicycle repairs | 2 | 0 | 0 | Wheel alignment and crank coupling |
| Skateboard repair | 1 | 0 | 0 | Wheel/axle connection |
| Microwave repairs | 2 | 0 | 0 | Door hinge and dish pivot |
| Car-door repairs | 2 | 1 | 0 | Beetle output truncated at token cap |
| TV/laptop pivots | 2 | 0 | 0 | Overlap and disconnected joints |
| Previous Camry pipeline failures | 2 | 0 | 0 | One geometry failure, one first-call timeout |
| **Main suite** | **20** | **2** | **2** | **4 total passes within the feedback budget** |
| Supplemental single-axle controls | 4 | 1 | 1 | Shifted 6-stud and 8-stud cases |

The main suite has 19 first responses, 18 parsed edit lists and one first-call service timeout. The feedback arm recovers 2 of the 17 failures with a response; generic retry recovers zero. The saved-mesh Camry generic retry also times out. Thus the 4/20 end-to-end diagnostic completion count includes unavailable service/output cases and must not be presented as a pure geometry score. The fresh-Camry case remains unmeasured in this run; its prior geometric failure is documented separately in the photo-pipeline audit.

The single-axle suite recovers 1 of 3 failures with feedback and zero with generic retry, yielding 2/4 total passes. Main and supplemental r3 traces record 26,795 and 2,730 generated tokens respectively. These totals exclude usage from interrupted requests and do not provide an exact bill.

## Concrete failure patterns

- **Cars:** first answers use 4-stud supporting beams even when the end supports require 6 or 8 studs. On joint tasks the model also omits the second axle. Feedback completes the simple 4-stud car; on the longer-span cars it adds the second axle but retains short beams. The shifted 6-stud case fails too.
- **Previous photo failures:** the saved-mesh Camry again receives short rear beams and fails after feedback. Its generic retry and the fresh-Camry first call time out; those outcomes do not establish a new geometric verdict.
- **Single-axle controls:** the 4-stud case passes immediately. Feedback repairs the unshifted 6-stud case by selecting two 6-stud beams. Neither the shifted 6-stud case nor the 8-stud case recovers. A generic retry fixes none. This rules out the strongest claim that the model can *never* resize a beam, while exposing inconsistent transfer under translation.
- **Train:** both four- and six-axle tasks receive one axle assembly with 4-stud beams. Feedback repeats the same edits. These combine a task-completeness failure with a support-span failure. The valid procedural references are rigid rolling chassis with a small locomotive silhouette: rail gauge, flanges, curves, bogies, couplers and propulsion are outside the test.
- **Bed:** the model adds repeated tall pieces above the bed and outside its target bounds instead of its eight missing leg bricks. Feedback and the generic retry still leave four ground supports absent. This is a failure to act on the requested support task, not a missing catalog part.
- **Chair:** the seat-pivot repair passes first try. The wheel-height repair passes after verifier feedback but not after the generic retry.
- **Bicycle:** both wheel alignment and crank attachment fail in every arm. The first answers to these different defects edit the same last three parts, rather than the affected wheel or crank. The feedback answers still fail the original mechanism requirements.
- **Skateboard:** the first response violates the requirement to preserve part identity/fields. Corrections become syntactically acceptable but leave the wheel disconnected and fail rolling/spinning checks. Improved instruction compliance alone is not a functional repair.
- **Microwave:** the door-hinge response re-emits broken hinge coordinates. The dish-pivot task edits door hinges instead of fixing the turntable. Feedback does not restore the required payload motions.
- **Doors:** the Camry hinge repair passes immediately. The Beetle response reaches the 4,000-token limit and produces an incomplete edit list in all three arms; this is an output/format failure, so its geometry cannot be scored.
- **TV/laptop:** attempted repairs introduce overlaps and leave the required screen joint disconnected. Feedback repeats the unsuccessful edits.

## What the training data supports

The reproducible offline inventory of `data/running_gear_frame/train.jsonl` finds 3,525 rows, including 1,169 feedback rows. In final assistant edits, 12,864 of 17,010 Technic beam instances are 4 studs long (75.6%). The remaining counts are 2,447 six-stud and 1,699 eight-stud instances.

Of those 1,169 feedback rows:

- **Zero** resize a beam at the same geometric center and y/z position.
- **Zero** change the beam-width multiset while preserving the number of beams.
- **Zero** feedback messages contain “longer.”

Although 251 feedback rows change the width multiset, they also change the beam count. Those are not evidence of supervision for resizing an existing beam. The r2 dataset has 12 same-center resizing rows; the older original/open datasets have none. Counts describe the local training files, not a complete reconstruction of checkpoint training history or proof that this imbalance caused the observed behavior.

The evidence suggests a gap between the correction behavior needed here and the corrections represented in r3's local supervised data. The successful single-axle repair also shows why the conclusion should be “unreliable generalization,” not “incapable of using feedback.” Cross-family repair tasks are transfer probes outside the narrow running-gear task distribution.

## Next implementation and training experiment

1. **Teach targeted repairs:** generate verifier-backed wrong-beam/right-beam pairs with fixed centers, balanced 4/6/8 lengths, translations, both axle orientations, and guards that reject overlong beams. Preserve the rest of the object. Include traces where the original failure disappears without introducing a new one.
2. **Cover each mechanism:** corrupt wheel heights, axle retention, crank coupling, hinge pins, turntable alignment, support feet and payload clearance. Train against the affected part IDs, with original numbering and all unrelated requirements preserved. Include irrelevant-edit and no-op negatives.
3. **Compose multiple tasks:** progress from one axle to two, four and six. Evaluate sequential construction as well as joint completion. Require a final whole-object check even when individual stages pass; never substitute reference pieces for failed learned steps.
4. **Check feedback causally:** compare feedback against the same-budget generic retry across multiple seeds. Split by source object/template before perturbation, hold out translations and spans, and retain these diagnostic cases as development data rather than relabeling them held out.
5. **Gate broader photo claims:** after mechanism repairs generalize, run fresh-photo evaluations for each supported object family, including perception, body generation, learned edits and exports. The present probes hold perception/body construction fixed.

No new cloud training was started. These are concrete next experiments, not completed capability improvements.

## Checker and artifact limits

Running-gear proposals use the shared edit/check path and require every requested axle line. Repairs permit only `set`, preserve part identities and use each family's native structural and mechanism checks. Bed completion permits only `add` and checks its target, stability and all four corner supports.

For screen/door payloads, the audit binds motion to the reference's intended part group and permits either side of the appropriate joint. This avoids the existing general helper's smallest-side ambiguity and vertical-axis assumption. This adapter is local to the diagnostic; it is not a broad rewrite of production motion checking. Collision sweeps are sampled and the checks remain simplified geometric models. Real assembly access, tolerance and load validation remain open.

The gallery labels procedural references and learned proposals separately; animation is disabled. Reference train/bed instructions include every supported part but do not verify insertion order or temporary supports. These instructions are not learned successes. Historical saved-build inventory also distinguishes stale schemas/unsupported checker types from actual geometry failures; old snapshots should not be treated as current generation results.

An initially started broad base-model comparison was stopped after the object-scope clarification. Three completed base responses returned no edit text at the 4,000-token limit. A separate 8,000-token single-axle probe timed out after a 120-second response wait. These are infrastructure/output diagnostics, not a matched baseline result. Interrupted requests may have incurred service usage not represented in saved token counts.

## Reproduction and evidence

Use the repository Python environment for offline commands. Live evaluation requires the external Fireworks training runtime documented in [photo-pipeline.md](photo-pipeline.md). Preparation and replay make no inference calls.

```bash
python -m samgo.eval_objects --prepare --out runs/audit/new-object-cases
python -m samgo.eval_objects --out runs/audit/new-object-cases --models r3
python -m samgo.eval_objects --verify-saved --out runs/audit/new-object-cases
python -m samgo.eval_objects --summarize --out runs/audit/new-object-cases
python -m samgo.eval_objects --render --out runs/audit/new-object-cases
```

Add `--include-local-fixtures` when preparing to include the two historical doors and two failed Camry steps; requested missing fixtures raise an error. Use `--single-axle` when preparing a separate four-case control bundle. Frozen cases cannot be overwritten through the CLI; completed evaluations resume without another service call, and changed case/protocol/sampler identity rejects resume.

```bash
python -m samgo.audit_training data/running_gear/train.jsonl \
  data/running_gear_open/train.jsonl data/running_gear_r2/train.jsonl \
  data/running_gear_frame/train.jsonl --out runs/audit/training-inventory.json
```

Local ignored artifacts are under `runs/audit/object-diagnostics/` and `runs/audit/object-single-axle/`: frozen `cases.json`, oracle/input admission checks, protocol, raw responses and exact messages, applied candidates, per-arm errors, summaries and offline replay checks. Training inventory includes source-file SHA256 hashes. `visuals/gallery.html` displays the saved assemblies. Source and compact findings are committed; large local fixtures and live outputs remain ignored.

## Verification

The full repository suite passed **238 tests in 47.58 seconds**. The eight new audit tests also passed after final CLI/report additions. Ruff checks (`E4,E7,E9,F`) and whitespace checks pass. All 16 portable cases and four supplemental cases pass reference admission and reject their defective inputs. The evaluated 20-case fixture bundle also passed admission. Offline replay of **49 main and 10 supplemental parsed proposals** exactly reproduces saved verdicts, error codes and candidate assemblies.
