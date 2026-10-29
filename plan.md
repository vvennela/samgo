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

## Costs and risks

- SFT: free on Fireworks until Nov 30. Serving the fine-tuned model needs an on-demand deployment
  (dedicated GPU, billed per hour); used only for eval and the demo, a few hours. Captions on Baseten.
- Fireworks account and API key needed (`FIREWORKS_API_KEY`).
- Risk: brick-by-brick decoding over an API is many small calls; prefix caching on a dedicated
  deployment keeps it to about a minute per build (BrickGPT: 41 s median). Fallback: vLLM with the
  LoRA on a rented GPU.
- Risk: SFT queue or a failed job. Fallback: same JSONL on RunPod with our LoRA trainer.
