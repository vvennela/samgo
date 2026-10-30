# GRPO RL (LoRA) on a $100 budget: platform and cost options (as of 2026-10-29)

Workload used throughout the worked estimates: GRPO, 16 prompts/step x 8 samples = 128 rollouts/step, prompt P = 2k-4k tokens (mid 3k), completion C = 300-800 tokens (mid 550). No KL-to-reference forward pass is assumed (Dr.GRPO-style); adding one costs roughly one extra prefill of every sequence.

## 1. Fireworks Training API: GRPO with LoRA on Muse Glimmer 30B, prices, and the promotion

### Takeaway
Yes. The Serverless Training API supports RL/GRPO with LoRA on `accounts/fireworks/models/muse-glimmer-30b`, and the cookbook has a ready Glimmer GRPO command (DeepMath). The four per-token prices you listed are confirmed on the live pricing page. RL is NOT covered by the free SFT promotion, which is LoRA SFT only. At these prices $100 buys about 22-48 GRPO steps (about 3k-6k rollouts) of the stated workload, and the train meter is about 80% of the cost.

### Cited Findings
- Serverless Training API rates for Muse Glimmer 30B (128K context), per 1M tokens: Prefill $1.96, Cached Prefill $0.39, Sample $4.88, Train $5.86. "Sample" is generation; "Train" is forward/backward. — [Fireworks pricing](https://fireworks.ai/pricing) (fetched 2026-10-29)
- Serverless docs: Serverless Training supports RL including GRPO-style training ("This is the primary serverless use case"). "Serverless is LoRA only." Meters: prefill (sampler prompt tokens), cached prefill (repeated prefixes, "an 80% discount"), sample (rollout tokens), train ("tokens passed through `forward_backward`"). No minimum spend is listed. Checkpoint storage is currently free. Default is 8 active runs per account. Some pool models also cap active runs per base model (`RESOURCE_EXHAUSTED`). The docs give no max LoRA rank; the examples use rank 8. — [Fireworks Serverless Training docs](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- Promotion: "Muse Glimmer 30B SFT is free via UI and Serverless Training API until 11/30." The pricing page covers SFT only and does not mention DPO or RL. — [Fireworks pricing](https://fireworks.ai/pricing)
- The cookbook skill's cost reference says the promotion covers LoRA SFT on Managed SFT and the Serverless Training API for jobs started before `2026-12-01T00:00:00-08:00`. It "does not cover full-parameter SFT, RL, DPO, Managed RFT, Dedicated trainers, deployments, inference outside the training run." It also says to label in-session evaluation sampling as a separate line. — fw-ai/cookbook `skills/configure/references/cost-estimation.md` (local plugin copy v2.3.0, `~/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0/`); mirrors [docs.fireworks.ai/fine-tuning/models](https://docs.fireworks.ai/fine-tuning/models)
- The cookbook Muse Glimmer README, step 4, is titled "RL on DeepMath (Serverless Training API, paid)": "RL is not covered by the promotion; price the run before launching it." Command: `python -m examples.rl.deepmath.train_deepmath_serverless --base-model accounts/fireworks/models/muse-glimmer-30b --tokenizer-model meta-models/Muse-Glimmer-30B --renderer-name muse_glimmer --max-rows 32 --prompt-groups-per-step 8 --completions-per-prompt 4 --max-completion-tokens 4096 --max-seq-len 8192 --lora-rank 8`. — fw-ai/cookbook `training/examples/muse_glimmer/README.md` (local copy v2.3.0; upstream [github.com/fw-ai/cookbook](https://github.com/fw-ai/cookbook))
- The cookbook's cost skill routes all Training API RL estimates to "contact the Training team" ([fireworks.ai/contact-training](https://fireworks.ai/contact-training)) rather than computing them. There is therefore no official RL cost formula; the arithmetic below is ours. — local `cost-estimation.md`
- "Muse Glimmer public serverless inference was deprecated on 2026-09-25." A tuned Muse LoRA can be sampled in-session on the Serverless Training API, or served from a paid on-demand deployment. — local `skills/research/references/starter-jobs.md`
- Managed (non-API) training tiers, per 1M training tokens: 16.1B-80B LoRA SFT $3.00, LoRA DPO $6.00. RFT/RL is not priced on this page. — [Fireworks pricing](https://fireworks.ai/pricing)
- Dedicated Training API uses on-demand GPU-hour rates: H100 80GB $8.00, H200 $8.00, B200 $13.00. — [Fireworks pricing](https://fireworks.ai/pricing)
- No Meta Llama 3.x model appears in the Serverless Training API table. That table lists Muse Spark Open, DeepSeek V4.1 Flash, GLM 5.3 (and Flash), Qwen 3.8 27B, Kimi K3, DeepSeek V4 Flash, and Muse Glimmer 30B. — [Fireworks pricing](https://fireworks.ai/pricing)

### Inferences
Worked estimate for Glimmer on Fireworks serverless. Assumptions:
- Prefill: 1 uncached plus 7 cached prefills per prompt group, because the 8 samples share the prompt prefix.
- Generation: the sampled tokens are billed at the Sample rate.
- Train: forward_backward is billed on the full prompt+completion sequence for all 128 rollouts. Prompt tokens pass through the model even when masked from the loss, and padding is not counted.
- No reference-model pass.

| Case (P, C) | Uncached prefill | Cached prefill | Generation | Train | $/step | Steps per $100 | Rollouts per $100 |
|---|---|---|---|---|---|---|---|
| Low (2k, 300) | 16x2k = 32k tokens x $1.96/M = $0.063 | 112x2k = 224k x $0.39/M = $0.087 | 128x300 = 38.4k x $4.88/M = $0.187 | 128x2,300 = 294.4k x $5.86/M = $1.725 | $2.06 | ~48 | ~6,200 |
| Mid (3k, 550) | 48k x 1.96 = $0.094 | 336k x 0.39 = $0.131 | 70.4k x 4.88 = $0.344 | 454.4k x 5.86 = $2.663 | $3.23 | ~31 | ~3,960 |
| High (4k, 800) | 64k x 1.96 = $0.125 | 448k x 0.39 = $0.175 | 102.4k x 4.88 = $0.500 | 614.4k x 5.86 = $3.600 | $4.40 | ~22 | ~2,900 |

- If prompt caching never hits, all 128xP prompt tokens bill at $1.96. The mid case then becomes 384k x 1.96 = $0.753 for prefill, and the step costs about $3.76 (about 26 steps).
- Train is about 82% of every step. The lever that matters is shorter prompts (each prompt is trained 8 times) or fewer samples per prompt, not shorter outputs. Moving the long, fixed context into a short prompt is the biggest saving. For example, cutting P from 3k to 1k in the mid case drops train to 128x1,550 = 198k tokens x $5.86/M = $1.16, and the step to about $1.6 (about 60 steps).
- Each step also has a sampling-only eval cost of about $0.0003 per eval prompt-token at the prefill rate plus $4.88/M for generated tokens. This is not in the table.
- About 30 GRPO steps on 16-prompt batches is a small RL run: 480 prompts seen once. Whether this is enough depends on the reward signal. Earlier SAMgo pilots found verifier-feedback retries beat best-of-k, which suggests the signal is learnable.
- Dedicated at $8/GPU-hr gives 12.5 H100-hours for $100. That is roughly comparable or worse for a 30B model, and there are minimum shapes. Treat it as not cheaper.

### Gaps
- Whether `n=8` samples from one request bill the prompt once or per sample. The docs only say "repeated prefixes hit the cache."
- Whether the train meter counts padded tokens for RL batches. The cost skill notes that serverless bills padded policy sequences for DPO, so RL may cost more than the table shows.
- Whether the GRPO recipe makes an extra forward pass for old or reference logprobs, which would add prefill or train tokens. Check `train_deepmath_serverless.py` before launch.
- Per-account Glimmer run-slot cap: project memory says there is 1 Glimmer slot, so always delete sessions.
- The pricing page does not give the year for the 11/30 promotion end. The cookbook pins it to 2026-12-01T00:00-08:00.

## 2. Tinker (Thinking Machines Lab)

### Takeaway
Tinker no longer offers any Meta Llama model: the Llama 3.x launch lineup was retired in June 2026. The closest small dense substitute is Qwen3-8B, the cheapest small dense model on the page, at about $0.27 per step for the mid case, or about 380 steps per $100. That is about 12x more steps than Glimmer on Fireworks. Tinker trains LoRA adapters, and access is via a "Join here" signup with no waitlist mentioned.

### Cited Findings
- Price table "as of October 19, 2026", per 1M tokens (prefill / cached prefill / sample / train):
  - Qwen3-8B (32K): $0.195 / $0.039 / $0.60 / $0.44
  - Qwen3.5-4B (64K): $0.33 / $0.066 / $1.005 / $0.737
  - Qwen3.5-9B (64K): $0.30 / $0.06 / $0.807 / $1.463
  - GPT-OSS-20B (32K): $0.18 / $0.036 / $0.45 / $0.396
  - Nemotron-3.5-Lightning-30B-A3B (64K, 50% off): $0.184 / $0.0368 / $0.495 / $0.44
  - Qwen3.6-35B-A3B (64K): $0.45 / $0.09 / $1.335 / $1.177
  - Qwen3.8-27B (64K): $0.6975 / $0.1395 / $5.595 / $4.103

  Cached prefill is "80% less". Storage is $0.10/GB/month. No Llama models, free credits, or minimums are listed. — [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/)
- Tinker "retired its entire launch lineup (the Llama-3.x family, Qwen3-32B, Qwen3-235B) in June 2026". There was a price increase on July 17, 2026: Qwen3-8B train went from $0.40 to $0.44 and sample from $0.40 to $0.60. — [Beam blog (third-party)](https://www.beam.cloud/blog/tinker-model-pricing)
- Conflict: Beam lists Qwen3.5-9B sample at $1.995 after July 17, but the live Tinker page lists $0.807. Use the live page. — [Beam](https://www.beam.cloud/blog/tinker-model-pricing) vs [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/)
- Old January 2026 Llama-3.1-8B rates (prefill $0.13, sample $0.40, train $0.40) appear only on third-party skill listings. They are obsolete because the model was retired. — e.g. [skills.sh tinker-training-cost](https://skills.sh/sundial-org/skills/tinker-training-cost)
- The Tinker page says "Model-agnostic LoRA training from 1B to 1T+ parameters" and lists "SFT, RL (GRPO, PPO), DPO, distillation". The FAQ says Tinker uses LoRA. Access is via "Join here", with contact for wide-scale university or org access. No waitlist is mentioned. — [Tinker](https://thinkingmachines.ai/tinker/)
- Model catalog: 28+ models, including Qwen3-8B, Qwen3.5-4B/9B, GPT-OSS-20B/120B, Nemotron variants, DeepSeek-V4.1-Flash, Kimi-K2.6, and GLM-5.3. Some models were marked "retiring October 23". — [Tinker](https://thinkingmachines.ai/tinker/)

### Inferences
Same workload and caching assumptions as section 1:

| Model, case | Uncached prefill | Cached prefill | Generation | Train | $/step | Steps per $100 |
|---|---|---|---|---|---|---|
| Qwen3-8B, Low (2k, 300) | 32k x 0.195 = $0.006 | 224k x 0.039 = $0.009 | 38.4k x 0.60 = $0.023 | 294.4k x 0.44 = $0.130 | $0.168 | ~600 |
| Qwen3-8B, Mid (3k, 550) | 48k x 0.195 = $0.009 | 336k x 0.039 = $0.013 | 70.4k x 0.60 = $0.042 | 454.4k x 0.44 = $0.200 | $0.265 | ~378 (~48k rollouts) |
| Qwen3-8B, High (4k, 800) | 64k x 0.195 = $0.012 | 448k x 0.039 = $0.017 | 102.4k x 0.60 = $0.061 | 614.4k x 0.44 = $0.270 | $0.362 | ~276 |
| Qwen3.5-4B, Mid | $0.016 | $0.022 | $0.071 | $0.335 | $0.444 | ~225 |
| Qwen3.8-27B, Mid (Glimmer-size reference) | $0.033 | $0.047 | $0.394 | $1.864 | $2.34 | ~43 |

- On Tinker the 4B model is more expensive per step than Qwen3-8B, because its per-token rates are higher. Qwen3-8B or GPT-OSS-20B are the cheapest options.
- A 27B-class dense model costs about the same per step on Tinker as Glimmer 30B on Fireworks ($2.34 vs $3.23). Size, not platform, drives cost.

### Gaps
- Whether the Tinker sampler bills the prompt once for `num_samples=8` (likely, through the prompt cache, but not verified).
- No Meta Llama models are available on Tinker. A Llama-specific comparison on Tinker is not possible.

## 3. Other managed RL services

### Takeaway
Few per-token GRPO services exist besides Fireworks and Tinker. CoreWeave/W&B Serverless RL (built on OpenPipe ART) is the notable one: training is free during public preview and you pay only inference rates. Its supported base models, and whether any Llama is among them, were not verified. Together prices RL only as GPU-hours. No verified Prime Intellect RL price was found.

### Cited Findings
- W&B Training: training is in public preview and free; pricing will be announced at general availability. Inference is billed at Serverless Inference rates, for example Qwen3 30B A3B at $0.10/M input and $0.30/M output. Storage is billed by checkpoint size, with some free storage included. — [W&B Training pricing](https://wandb.ai/site/pricing/training/)
- "Training is free during the public preview period." "Pricing for Serverless Training inference requests matches Serverless Inference pricing." Defaults are 2,000 concurrent requests per user and 6,000 per project. It is available only in supported regions. (The docs moved to CoreWeave.) — [CoreWeave Serverless RL usage limits](https://docs.coreweave.com/training/serverless-rl/usage-limits)
- Together AI: Llama 3.1 8B fine-tuning is $0.34-0.38/M for SFT and $0.84-0.94/M for DPO, with a $4 minimum (two tables, LoRA vs full not labelled). No RL/GRPO per-token product is listed. "Post-Training" is per GPU-hour: B200 $9.00/hr (promo until 12/14/2026; list $11.00). Llama 3.2 3B is not listed. — [Together pricing](https://www.together.ai/pricing)
- Prime Intellect publishes open-source Verifiers and prime-rl (MIT) and sells GPU-hours. No managed-RL price was found. Aggregator pages say pricing visibility is limited. — [airframe.ai comparison](https://www.airframe.ai/product/primeintellect-ai/compare); [Prime Intellect Series A](https://www.primeintellect.ai/blog/series-a)

### Inferences
- If W&B/CoreWeave Serverless RL supports a suitable small model and the SAMgo reward can run inside ART's rollout loop, it may be the cheapest managed route: $100 would pay only for inference tokens. The mid case is about 0.45M input + 0.07M output tokens per step, a few cents at Qwen3-30B-A3B rates, so the step count would be bounded by preview limits, not money. This is speculative until the model list and preview terms are checked.

### Gaps
- Supported base models for W&B Serverless RL (whether any Llama 3.x is included) and the preview end date.
- OpenPipe standalone pricing (it now appears folded into CoreWeave/W&B).
- Prime Intellect hosted RL pricing; Hugging Face managed GRPO (none found).

## 4. Raw GPU rental and GRPO throughput (Llama 3.2 3B / Llama 3.1 8B with TRL/Unsloth/verl)

### Takeaway
RunPod Community is the cheapest listed on-demand option: H100 SXM $2.69/hr, A100 80GB $1.39/hr, L40S $0.79/hr, RTX 4090 $0.34/hr. Modal costs about 1.5x more but bills per second and includes $30/month of free credit. No credible published GRPO-with-vLLM throughput figure was found for Llama 3.1 8B or 3.2 3B. The step counts below rest on our own FLOP estimate, roughly 2-5 minutes per mid-case step for 8B on one H100. On that basis $100 buys about 400-1,100 steps of 8B, and more for 3B. Expect real losses to setup and debugging.

### Cited Findings
- RunPod, per hour, Community / Secure / Serverless:
  - H100 PCIe: $1.99 / $2.89 / $5.39
  - H100 SXM: $2.69 / $3.99 / $5.39
  - A100 PCIe 80GB: $1.19 / $1.79 / $2.72
  - A100 SXM 80GB: $1.39 / $1.79 / $2.72
  - L40S: $0.79 / $1.09 / $1.75
  - RTX 4090: $0.34 / $0.89 / $1.34
  - RTX 5090: $0.69 / $1.19 / $1.79
  - H200: $3.59 / $5.29 / $6.99

  The community/secure split comes from the page's structured data. — [RunPod pricing](https://www.runpod.io/pricing)
- Modal, per second (per-hour figures are our conversion): H100 $0.001097 (~$3.95/hr), A100 80GB $0.000694 (~$2.50/hr), L40S $0.000542 (~$1.95/hr), L4 ~$0.80/hr, H200 ~$4.54/hr, B200 ~$6.25/hr. The Starter plan includes $30/month of free credit. — [Modal pricing](https://modal.com/pricing)
- Lambda H100 SXM on-demand is about $3.99/GPU-hr (8-GPU node); single H100 PCIe about $3.29. Lambda bills in full-hour increments. These are third-party figures, mid-2026, not verified on lambda.ai. — [Spheron](https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/); [computeprices](https://computeprices.com/providers/lambda/gpus/h100)
- Vast.ai H100 SXM marketplace prices are typically ~$1.35-2.21/hr, varying by host, with no SLA. These are third-party, mid-2026 figures. — [IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison); [Keenable](https://select.keenable.ai/r/effective-nvidia-h100-cloud-prices/0DRsK7gE-5uMsmQ7IiUM1A)
- Unsloth's benchmarks claim about 2x speed and over 70% less VRAM vs HF+FA2 for Llama 3.1 8B (QLoRA r=32). They give no GRPO or vLLM rollout figure. — [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)

### Inferences
Our estimate (not sourced) for Llama-3.1-8B with LoRA on 1x H100, mid case:
- Training pass: about 4 x 8e9 x 454k ≈ 1.5e16 FLOPs for LoRA forward+backward (no weight gradients), plus about 1/3 for gradient-checkpoint recompute, giving about 2e16. At about 35-40% MFU (about 350-400 TFLOPs) this takes about 50-60 s. A logprob pass adds about 15 s.
- vLLM rollouts: 128 sequences, about 70k generated tokens. With the long tail to 800 tokens this takes about 30-60 s. Prefix-cached prefill takes a few seconds.
- Weight sync and overhead in a colocated TRL/Unsloth setup: 10-60 s.
- Result: about 2-3 min/step with good tooling (verl or Unsloth+vLLM), and up to about 5 min with plain TRL.

| Option | Hours per $100 | Steps per $100, mid case |
|---|---|---|
| 8B, RunPod Community H100 SXM ($2.69) | 37 h | ~450-1,100 (at 5-2 min/step) |
| 8B, Modal H100 ($3.95) | 25 h | ~300-750 |
| 8B, RunPod Secure H100 ($3.99) | 25 h | ~300-750 |
| 3B, A100 80GB at $1.39 (Community) | 72 h | ~2,000-4,000 (at about 1-2 min/step) |
| 3B, L40S at $0.79 | ~126 h | similar range, more VRAM-constrained |

- For 3B, roughly 2.5x fewer FLOPs per step than 8B. A 24GB RTX 4090 is tight for 128 x 3.5k-token rollouts plus training, so 4-bit Unsloth with small micro-batches would be needed.
- Realistically, budget 20-30% of the hours for environment setup, model download, debugging and evals.
- In steps per dollar, raw GPUs on a small Llama give roughly 15-35x Glimmer-on-Fireworks and roughly 1-3x Tinker Qwen3-8B. The cost is the engineering time to run vLLM+GRPO yourself, and the AWS-style babysitting rules (shut down after, check every 10 min).

### Gaps
- No primary source found for GRPO end-to-end step time or tokens/s with vLLM on Llama 3.1 8B / 3.2 3B. The numbers above are our estimates and should be checked with a 5-step smoke run.
- Lambda and Vast prices were not fetched from the vendors' own pages. The figures are mid-2026 aggregator data.
- Whether Llama 3.x Instruct is a sensible base for the task compared with Qwen3-8B is out of scope here.

## 5. Bottom line ($100 cap)

### Takeaway
For the mid-case workload, $100 buys:

| Option | Steps per $100 | Rollouts per $100 |
|---|---|---|
| Glimmer 30B on Fireworks serverless (RL is paid; promotion is SFT only) | ~31 (22-48) | ~4k (3k-6k) |
| Qwen3-8B on Tinker (no Llama available there) | ~380 (276-600) | ~48k |
| Qwen3.8-27B on Tinker | ~43 | ~5.5k |
| Llama-3.1-8B on a RunPod H100 (unverified throughput) | ~450-1,100 | — |
| Llama-3.2-3B on an A100 80GB | ~2,000+ | — |
| W&B/CoreWeave Serverless RL | free training during preview, inference billed (models unverified) | — |

### Cited Findings
- See sections 1-4 for every rate used: [Fireworks pricing](https://fireworks.ai/pricing), [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/), [RunPod pricing](https://www.runpod.io/pricing), [Modal pricing](https://modal.com/pricing), [W&B Training pricing](https://wandb.ai/site/pricing/training/).

### Inferences
- On Glimmer, the free SFT promotion through Nov 30 Pacific makes SFT-style alternatives free. Examples: rejection-sampling SFT on verifier-passing samples, or SFT on retry traces. The $100 then pays only for sampling. GRPO on Glimmer gets about 30 steps, so it is best kept as a small, final polishing stage with short prompts. Cutting P from 3k to 1k roughly doubles the step count.
- Note the project's own prior result: SFT on self-samples was NO-GO twice. Free SFT is cheap but has not worked here before.

### Gaps
- The official Fireworks RL cost path is "contact the Training team". The serverless arithmetic above is ours and excludes padding and possible extra logprob passes.
