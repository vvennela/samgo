# SAMgo - working rules for Claude

Photo -> LEGO model that works like the real thing. Read `docs/HANDOFF.md` for the current state and numbers.

## The point (read first)

**We are training a general builder of working machinery, not building particular objects.** The goal is that any
photo - a car, a fan, a turbine, a satellite - becomes a LEGO model whose mechanisms work, built by the same models.
Generality across all machinery is what wins.

## Architecture: Spark knows, Glimmer does, code verifies

```
photo -> SAM 3.1 (mask) -> SAM 3D (mesh) -> code: mesh -> LEGO body (voxels -> plates/bricks)
                                                   |
  SPARK (Muse Spark Open, Baseten)   GLIMMER (Muse Glimmer 30B + our LoRA,   CODE (verifiers)
  knows: names the parts and the     Fireworks) does: gets a local handoff   checks every answer: valid parts, no
  moving parts of the real object    (nearby parts + one sentence) and       overlap, every part held by studs (joint
  from the photo, then points where  writes the edit list (clear / remove /  halves built in), the mechanism moves
  each goes on the model. World      set / add). The only thing that builds.  clear, stands, buildable step by step;
  knowledge only, no hints.                                                  sends precise feedback, up to 3 rounds
```

- **Feedback** = the checker's failures turned into exact sentences naming the parts and the fix ("input part 86
  overlaps your bar plate: remove it first"; "this clip holds nothing"). Glimmer retries with it. Training includes
  mistake -> feedback -> fix rows so Glimmer learns to use it.
- Spark never places a brick; Glimmer never decides what the object is; code never builds what Glimmer failed to
  build (that mechanism is left out and reported).
- Key files: `samgo/pipeline.py` (photo -> model), `samgo/spark_mechanisms.py` + `samgo/part_layout.py` (Spark names
  and grounds mechanisms), `samgo/spark_gear.py` (legs/wheels), `samgo/free_edit_test.py` (`check`, `explain` =
  feedback), `samgo/kinematics.py` (joints, motion), `samgo/manual.py` (certified build steps), `samgo/edits.py`.

## No hard coding

- **Nothing object-specific in code, data or prompts.** No `if airplane`, no plane-shaped templates, no aircraft
  words in training data or Spark prompts. The test object's name is passed to Spark only (it may use its own
  knowledge); Glimmer's handoffs use generic wording ("one edge of a flat structure").
- The 747, F-16 and any rocket, satellite, Cessna... are **held-out exams**. Never train or tune on them.
- When an exam exposes a gap, the fix is **a generic checker rule + generic training data so Glimmer learns it**.
  Never hand-patch the test object (no stands, no custom fixes so it passes). Fixing an eval bug is fine.
- Features apply wherever they make sense, not per object family: e.g. glass/panes wherever the real object is
  transparent (Spark says where), not a car-only window routine.
- When reporting, say whether anything was looked at or changed because of the test object.

## Verification is physical

Everything must be buildable in real LEGO. A check that passes something you could not build is a bug in the check.
Example (Nov 2): a clip hanging on its bar over empty space passed `connected`; now a joint half whose studs grip
nothing fails (`kinematics.dangling`), and data generated under the old check was regenerated.
Step-by-step buildability is certified exactly by `samgo/manual.py` (rational force balance per step). There is no
Lean proof; `docs/formal-verification.md` states the trust boundary.

## Rules

- Commit after each part; **never push**; **never merge or fast-forward into `main`**.
- **No Claude attribution anywhere**: no `Co-Authored-By: Claude`, no "Generated with Claude Code" / "done with
  Claude" trailers or lines in commit messages, PRs, or anything pushed to GitHub.
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

## Hackathon (Meta Global AI Developer Hackathon)

- **Deadline: Wed Nov 4, 9:00 PM ET (6:00 PM PT)**, submitted on the Cerebral Valley platform. Building period Oct 27 -
  Nov 4; this repo's history starts Oct 27, so everything here is new work.
- **Track:** Indie Developer. **Problem statement:** "Push the Frontier" - give machines new capabilities in the
  physical world (robotics, manufacturing...). Our pitch: Meta's models turning a photo into a working, buildable
  machine, generalizing to machinery never trained on.
- **Must use Meta models:** we use SAM 3.1 (Meta Model API), Muse Spark Open (Baseten, model
  `meta-models/muse-spark-open-1026`) and Muse Glimmer (open weights, fine-tuned on Fireworks).
- **Submission needs:** a **public** repo (the user pushes; check it is public), a **3-minute demo video** showing
  only what was built during the hackathon, everyone who worked on it listed.
- **Disqualifiers:** assets or data we lack rights to (record photo sources in `photos/SOURCES.txt`); anti-project
  topics, including **anything political or involving war**. Keep military objects (fighter jets, missiles) out of the
  demo and submission; use civil machinery (airliners, cars, fans, turbines, appliances).
- **Judging:** pass/fail on theme and real use of Meta models, then a 0-10 rubric, then top 5 per track judged live on
  the video. The video and the code are the main artifacts.
