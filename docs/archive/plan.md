> **Superseded (Oct 29):** history only. The current plan is `spec.md` (repo root) section 0.

# SAMgo build plan: replicate BrickGPT with a Meta model, then make it work

**Goal.** The model is the builder. Muse Glimmer places every brick, one at a time, in a real 3D grid; a
physics check catches what won't stand and the model rebuilds from there. Output is an LDraw file a person
(or BrickLink Studio) can build from. Then we extend the one thing no paper does: builds that *work* (wheels
that roll).

**Paper we replicate.** BrickGPT (Pun et al., ICCV 2025 oral, arXiv 2505.05469; code, model and data
released). Their recipe and numbers are our targets.

| | BrickGPT | SAMgo |
|---|---|---|
| Builder model | LLaMA-3.2-1B, LoRA r32 on q/v | **Muse Glimmer 30B**, LoRA on Fireworks (free SFT until Nov 30) |
| Data | StableText2Brick, 42.6k train / 4.8k test, MIT | same dataset, same split |
| Format | one brick per line, `hxw (x,y,z)`, 8 bricks, 20x20x20 grid, 1 unit = 1 brick tall | same |
| Prompt | "Create a LEGO model of {caption}" | same, plus the brick library in the system prompt |
| Decoding | brick-by-brick rejection sampling (malformed, not in library, out of bounds, collision → resample, temp 0.6 + 0.01 per reject) | same, with our structural verifier (S3a) |
| Physics | at the end: stability analysis; if unstable, roll back to just before the first unstable brick and continue; ≤ 100 rollbacks | same, with our force-balance LP (S3b) |
| Color | nearest standard color, after the fact | same |
| Output | text, rendered via LDraw | **LDraw `.ldr` with real part IDs**, open in BrickLink Studio for instructions |
| Their result (250 test prompts) | 100% valid, 98.8% stable; base LLaMA 5-shot 2.4% valid; no rollback 24% stable | target: ≥ 95% valid, ≥ 90% stable |

## What the demo shows

Photo → Muse Spark describes it (shape caption + colors) → **Glimmer builds it brick by brick, live**:
each brick appears in the 3D view, rejected bricks flash and are resampled, an unstable result rolls back
and is rebuilt. → LDraw file + instructions → the physical build.

## Steps

1. **Data → JSONL** (Thu). StableText2Brick train split to Fireworks chat JSONL, one caption per
   structure per row (42.6k rows; a second caption per structure if time allows).
2. **Baselines** (Thu, parallel). On BrickGPT's 250 test prompts with the same decoding: Glimmer zero- and
   few-shot, Spark Open few-shot. Expected near BrickGPT's untrained baseline (≈ 0-2% valid): that gap is
   the point of training.
3. **Fine-tune** (Thu-Fri). Fireworks SFT, Glimmer 30B, LoRA, 1 epoch, defaults otherwise. Free.
4. **Decoder** (Thu-Fri, while training). Brick-by-brick loop over the Fireworks completions API with
   Glimmer's chat template: generate one line, check it, resample or accept; at the end run S3b and
   roll back. Log every accept / reject / rollback (this log drives the live view).
5. **Replication eval** (Fri). Same 250 prompts and metrics as their Table 1: valid %, stable %, mean
   stability, rollbacks per build, time per build. Compare to BrickGPT's numbers.
6. **LDraw + instructions** (Fri). `.ldr` export: 1x1 3005, 1x2 3004, 1x4 3010, 1x6 3009, 1x8 3008,
   2x2 3003, 2x4 3001, 2x6 2456; 20 LDU per stud, 24 per brick, -y up. Verify in BrickLink Studio.
7. **Photo front end** (Sat). Spark Open writes a BrickGPT-style caption from the photo; same pipeline.
8. **Make it work: wheels** (Sat-Sun). Add Technic parts to the format (Technic brick with holes, axle,
   wheel). Training data: ShapeNet cars from StableText2Brick rebuilt with the algorithmic car builder,
   checked by the function verifier (S3c). Warm-start SFT from step 3. Rollback also triggers on S3c.
   Target: the car prompt yields a model that passes S3c (axles in round holes, wheels clear, wheels lowest).
9. **App + physical build** (Mon). Web page with the live build view; build one generated car by hand
   and film it rolling.
10. **Video + write-up** (Tue). Submit Wed 11/4.

## What we reuse, what we drop

- Reuse: structural verifier (S3a) = their validity check; stability LP (S3b) = their stability
  analysis; function verifier (S3c); SAM 3.1 / SAM 3D / Spark perception; instruction renderer as a fallback.
- The algorithmic builder only makes training data for the wheel extension (as BrickGPT's own
  dataset was made by a greedy builder). It is not the product.
- Dropped: verifier-only training, reverse curriculum, GRPO. Not needed for the demo; GRPO stays as
  future work.

## Fireworks (researched Oct 29)

- **Model:** `accounts/fireworks/models/muse-glimmer-30b` (29.7B dense, 131k context, Apache 2.0, image input,
  reasoning strength low..xhigh). Tunable with LoRA (rank up to 128, firectl default 8).
- **Training:** "Muse Glimmer 30B SFT is free via UI and Serverless Training API until 11/30", no card needed.
  List price otherwise $3 / 1M tokens (our ~65M tokens would be ~$195). Launch from the UI to be sure the
  promotion applies. Data: chat JSONL with `messages`, 3 to 3M lines; set `--lora-rank` explicitly (an omitted
  rank in the API means full-parameter training). Track loss with the W&B flags.
- **Serving:** no serverless for Glimmer after Sep 25; on-demand only, **$8 per H100- or H200-hour**, billed per
  GPU-second while replicas run. Scale-to-zero only after 1 h idle, so delete the deployment after each session
  (`firectl deployment delete`). `--preemptible` exists for eval.
- **Decoding fits:** `/v1/completions` takes a raw prompt string or token ids (no chat template applied: we render
  Glimmer's template ourselves), `stop` (up to 4), `n` up to 128, `logprobs`, `return_token_ids`. Prompt caching
  is automatic on exact prefixes, so each brick-by-brick call reuses the growing build; cache hits show in
  `usage.prompt_tokens_details.cached_tokens` and the `fireworks-cached-prompt-tokens` header.
- **The LoRA can be downloaded** (listed as `HF_PEFT_ADDON`, standard PEFT files), so we can train free on
  Fireworks and serve it ourselves with vLLM on a rented GPU (~$2.50/h) if $8/h adds up.
- Setup: `brew tap fw-ai/firectl && brew install firectl`, `firectl signin`; API calls use `FIREWORKS_API_KEY`.

## Costs and risks

- SFT: free on Fireworks until Nov 30. Serving: Fireworks on-demand at $8/h (eval + baselines about 3 h,
  ~$25; demo sessions extra), or download the LoRA and serve with vLLM on RunPod at ~$2.50/h. Captions on Baseten.
- Fireworks account and API key needed (`FIREWORKS_API_KEY`).
- Risk: brick-by-brick decoding over an API is many small calls; prefix caching on a dedicated
  deployment keeps it to about a minute per build (BrickGPT: 41 s median). Fallback: vLLM with the
  LoRA on a rented GPU.
- Risk: SFT queue or a failed job. Fallback: same JSONL on RunPod with our LoRA trainer.

## Status (Oct 29, 02:40)

| Piece | State |
|---|---|
| v1 data (BrickGPT format, verbatim prompts) | `data/sft/train_c2.jsonl` 85,208 rows, `eval_500.jsonl` |
| v1 fine-tune | launcher ready: `! bash brickgpt/start_finetune.sh` (Fireworks blocks agents from starting it) |
| Checkers in BrickGPT format | `brickgpt/structure.py`: 200/200 of their test structures valid, 193/200 stable (ours is stricter) |
| Decoder | `brickgpt/decode.py`: rejection + rollback, event log; real-API test 0.26 s per brick |
| LDraw | matches BrickGPT's exporter; v2 ids checked against the official library |
| 35-part palette | `samgo/palette.py`, 8 rule tests |
| v2 data | `data/v2/train_c2.jsonl` 77,124 rows (38,562 shapes, 90.5% kept), `test_c1.jsonl` 454 rows; ~1.9k tokens/row |
| v2 fine-tune | launcher ready: `! bash brickgpt/start_finetune_v2.sh` |
| Eval + render | `brickgpt/evaluate.py` (Table-1 metrics), `brickgpt/render.py` |
| Next | after training: serve (Fireworks $8/h or RunPod ~$2.50/h), 250-prompt eval, live demo view, Technic wheels |
