# SAMgo - working rules for Claude

Photo -> LEGO model that works like the real thing. Read `docs/HANDOFF.md` for the current state and numbers.

## The point (read first)

**We are training a general builder, not building particular objects.** Three roles:

- **Spark knows** (Muse Spark, Baseten): what the object is, what moves, where (its own world knowledge, no hints).
- **Glimmer does** (Muse Glimmer 30B + our LoRA, Fireworks): writes the edits that build each mechanism.
- **Code verifies**: structure, connection, motion, balance, buildability - and gives precise feedback. Code never
  builds the thing Glimmer failed to build.

The 747 (and any rocket, satellite, F-35, Cessna...) is a **held-out exam** of that builder:

- Never train or tune on a test object; no plane-shaped bodies, aircraft words or plane hints in data or prompts.
- When a test exposes a gap, the fix is **a generic checker rule + generic training data so Glimmer learns it**.
  Never hand-patch the test object (no stands, no custom fixes so the plane passes). Fixing an eval bug is fine.
- When reporting, say whether anything was looked at or changed because of the test object.

## Verification is physical

Everything must be buildable in real LEGO. A check that passes something you could not build is a bug in the check.
Example (Nov 2): a clip hanging on its bar over empty space passed `connected`; now a joint half whose studs grip
nothing fails (`kinematics.dangling`), and data generated under the old check was regenerated.
Step-by-step buildability is certified exactly by `samgo/manual.py` (rational force balance per step). There is no
Lean proof; `docs/formal-verification.md` states the trust boundary.

## Rules

- Commit after each part; **never push**; **never merge or fast-forward into `main`**.
- **No Claude attribution anywhere** (no `Co-Authored-By`, no "Generated with Claude") in commits, PRs, GitHub.
- Keys live in `.env`; never print them.
- Design before code on big changes (new skills, new training runs): propose, get the user's go-ahead.
- Background jobs via `run_in_background` / Monitor, never bare `&` in an interactive command.
- Share builds in 3D with `samgo/viewer3d.py` (not static renders).
- AWS only if asked. No license lectures.

## Fireworks

- Serverless LoRA SFT via `brickgpt/train_serverless.py` (SFT and sampling are free for us; one Glimmer training
  slot). Recipe: rank 32, batch 64, 2 epochs, max-seq-len 24576.
- **After a killed or stalled trainer, delete its training session** (snippet in `docs/HANDOFF.md` section 7),
  then rerun the chain; it resumes from the last checkpoint (every ~2,000 rows).
- Each run has `runs/rN_chain.sh` (data -> train -> tests, with a no-progress watchdog) and
  `fireworks-training-runs/objects-rN/run.md` (what, why, results).

## Testing

- Frozen test sets in `runs/audit/` are never regenerated. Current flap test: `r9-sheets` (`r8-sheets` retired: 12
  of its 30 references had floating clips).
- `eval_objects` caches per case under `results/<run>/`: move cases with `service_error` aside before re-running.
  Running many evals in parallel causes service errors; run object/door evals one after another.
- Eval commands need `PYTHONPATH=<repo>:<fireworks cookbook>` (see `runs/r9_tests.sh`).
- Unit tests: `UV_PROJECT_ENVIRONMENT=~/.venvs/samgo uv run pytest -q` (the repo `.venv` hangs on iCloud).
- iCloud makes "name 2.py" conflict copies that break pytest collection; move them out of the tree.
