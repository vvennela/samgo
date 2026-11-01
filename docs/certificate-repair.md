# Learned repair with certificate feedback

This experiment connects the existing running-gear r3 Glimmer checkpoint to the exact static certificate checker. It changes no model weights and uses no RL. It tests add-only repairs of supplied brick/plate objects, not generation from a photograph or certification of moving mechanisms.

## Repair contract

`samgo.certificate_repair` gives the model the original parts, catalog, coordinate units, addition budget, permitted support volume and certificate assumptions. The only permitted operation is `add`. Every attempt replaces the previous proposed additions and starts from the same original input. This prevents failed proposals from silently accumulating or replacing the intended object.

The checker validates edit syntax, catalog geometry, support bounds, final connectivity, non-overlap, continuous vertical insertion and exact force/torque equilibrium at every bottom-up placement prefix. A successful producer result must pass the independent rational checker. Only then can the command export a build, certificate and instructions preserving that certificate's order. The producer and instruction exporter both enforce the certificate gate.

Feedback includes the verdict, concrete issues, first uncertified part index and geometry, and completed prefix count when available. Invalid edit/bounds/geometry results can precede force solving. Feedback supplies constraints and observed failures, never a reference solution. Unsupported mechanisms do not become certified by stripping their parts.

Runs save prompts, raw answers, parsed edits, candidate assemblies, feedback, certificates, token counts, response-limit flags, sampler identity, code hashes and budgets. Sampler errors are separate from geometry failures and redact exception messages that could contain credentials. Completion/resume checks reject changed task, policy, protocol, sampler or code and recheck saved proposals. A request interrupted while pending is not automatically resubmitted because it might already have incurred inference usage.

## Controlled pilot

Eight development cases are frozen before inference: a six-stud shelf, an eight-stud translated shelf, a rotated shelf, a table, a bench, an assembly that needs support before its counterweight is placed, the original diagnostic bed and a translated bed. Every unedited input fails and every private reference repair passes the task and certificate checks. Reference repairs remain outside the sampler interface.

Each case has one shared first answer at temperature zero. When that answer fails, two paths each get up to two further responses at temperature 0.7:

- **Certificate feedback:** gets the previous proposal's task/certificate failure details.
- **Generic retry:** gets a request to reconsider all stated requirements, without checker results.

Both paths receive the same original task, failed first answer, token cap and response timeout. Call order is counterbalanced by case ID. There is no seed argument in the current sampler adapter, so this is a small stochastic pilot, not a matched-seed estimate of a causal feedback effect. Success stops that path early. First-response service errors are recorded and excluded from the feedback comparison rather than inventing a failed proposal.

The configured ceilings are three attempts per path including the shared first answer, 4,000 output tokens per call, 60 seconds for each sampling-result wait and 10 seconds per certificate producer invocation. Eight cases require at most 40 inference calls / 160,000 capped output tokens; this does not bound input-token billing or enforce a hard process-level deadline. Tokenizer/client initialization, checker setup and export can take additional time. Sampling uses the existing Fireworks account, not Prime RL credits.

## Reproduction

Use the repository environment for preparation, replay and summary. Live sampling requires the existing Fireworks training runtime and cookbook documented in [photo-pipeline.md](photo-pipeline.md), plus `z3-solver` (the pilot uses 4.16.0.0).

```bash
python -m samgo.eval_certificate_repair --prepare --out runs/audit/certificate-feedback-v2
python -m samgo.eval_certificate_repair --run --out runs/audit/certificate-feedback-v2
python -m samgo.eval_certificate_repair --replay --out runs/audit/certificate-feedback-v2
python -m samgo.eval_certificate_repair --summarize --out runs/audit/certificate-feedback-v2
```

Preparation refuses to overwrite frozen cases. `--case shelf_6` can run a smoke case; a later full run resumes it without resampling. `--rounds`, `--max-tokens`, `--timeout` and `--model` select an explicit protocol; use a new output directory when changing it.

For a single task JSON containing exactly `id`, `description`, `input`, `max_added` and `bounds` (`[xmin,xmax,ymin,ymax,maximum_top_z]`):

```bash
python -m samgo.certificate_repair task.json -o runs/my-repair --rounds 3
```

The new learned command is explicit. It does not silently substitute the procedural support search into a failed learned result or alter the existing photo pipeline's success definition. Preserving original geometry and limiting additions is a task-specific fidelity rule, not proof of semantic correctness, minimal part count, occupant load capacity or measured physical buildability. See [formal-verification.md](formal-verification.md) for the exact certificate scope.

The initial `certificate-feedback-v1` run was interrupted after discovering that the positional bounds array lacked a legend in the model prompt. Its partial traces and pending-request marker are retained for diagnosis and usage accounting; they are excluded from the final score. Version 2 uses the identical frozen cases with an explicit bounds legend and full-solid inequalities. No reference answer or observed solution was added to the prompt.

## Measured result — October 31, 2026

**The existing r3 checkpoint repaired none of the eight cases under this protocol.** This is a negative result for this model/prompt/repair task, not evidence that RL is necessary or that verifier feedback cannot work with a different proposal policy.

| Case | First attempt | With certificate feedback, ≤3 attempts | Generic retry, ≤3 attempts | Last feedback failure |
|---|---|---|---|---|
| Six-stud shelf | Fail | Fail | Fail | Duplicate original support; overlapping solids |
| Eight-stud translated shelf | Fail | Fail | Fail | Added support outside permitted bounds |
| Rotated shelf | Fail | Fail | Fail | Added support outside permitted bounds |
| Table | Fail | Fail | Fail | New supports overlap each other |
| Bench | Fail | Fail | Fail | No compression-only equilibrium at the seat-placement prefix |
| Counterweight prefix | Fail | Fail | Fail | Added support outside permitted bounds |
| Bed | Fail | Fail | Fail | New supports overlap each other |
| Translated bed | Fail | Fail | Fail | Added support outside permitted bounds |

All 40 scored calls returned responses: **12,676 output tokens, zero service errors and zero token-limit responses**. The first answer is counted once per case, not twice for the two paths. The 40 unique proposals comprise 21 support-bounds violations, two addition-budget violations, 12 solid-overlap failures and five uncertified compression-equilibrium results. These counts identify the first blocking check per proposal; they do not assert that later checks would pass.

Feedback retries repeated the immediately preceding edit list **11/16** times; generic retries did so **14/16** times. Changed answers sometimes fixed one defect while leaving another: the table eventually got past its bounds violation but its new supports still overlapped. On the bench, the model placed eight 1x1 support bricks across x but only in the y=0 row. Their contact points lie below y=1, while the two-stud-deep seat's center is at y=1; the checker rejects torque balance at step 9, before the back is placed. The model retained this arrangement after the equilibrium feedback.

Offline replay reproduces every saved verdict and exact candidate: **48 stored attempts, including eight duplicated shared-first records, corresponding to 40 unique calls**. All eight reference repairs were rechecked and pass; all eight original inputs still fail. No failed learned candidate was exported as an accepted build or given success instructions. The comparison gallery labels inputs, procedural references and rejected r3 proposals separately and disables motion animation.

The scored run is `runs/audit/certificate-feedback-v2/`: frozen cases, environment/code manifest, per-case first answers, both complete retry traces, summary, reference rechecks and `gallery.html`. A compact checked-in record with hashes, per-attempt failures and token accounting is [certificate-repair-results.json](certificate-repair-results.json). Its case hash is `29f4b196a885c6233ecbb186a344421432cfcc86e3ae14d079c6c4dd245d3cb8`.

For usage accounting, the superseded v1 run additionally recorded **24 completed responses / 7,106 output tokens**, with one sampling request pending at interruption that may have incurred further usage. Those responses are excluded from the score, not from the account's possible bill. Input-token usage and dollar charges are not provided by this adapter. No training or Prime resource allocation occurred.

Validation: the full local suite passed **307 tests** before the bounds-legend correction; after that prompt-only correction and its regression test, all **16 affected repair/evaluation tests** passed. Targeted Ruff and `git diff --check` pass. The full suite retains its existing local photo-fixture dependency.

The next useful no-training experiment is a more constrained proposal interface: give the model explicit legal placement choices, and make collision feedback name the offending original/addition geometries. Keep the same independent gate and frozen comparison. This is a proposed experiment, not an implemented capability or a promised improvement. The current result establishes that certificate enforcement rejects invalid output; it does not establish that supplying failure text alone teaches r3 general assembly repair.
