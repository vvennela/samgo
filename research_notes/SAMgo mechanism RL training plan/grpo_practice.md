# Small-budget GRPO (LoRA) on structured-edit tasks with verifiable rewards

Scope: recommended hyperparameters and procedures for GRPO on a task where an LLM emits a JSON list of edits to a 3D LEGO build, scored by a code verifier (pass/fail checks + continuous similarity to reference), after a short SFT warm-up, ~$100 budget, at most a few hundred steps.

## 1. Core hyperparameters (group size, prompts/step, steps, LR, LoRA rank, KL, clipping, entropy, loss aggregation, max length)

### Takeaway
For a small LoRA run, the consensus settings are: 8 samples per prompt (4 is acceptable if compute-limited); 16-64 prompts per step; LoRA LR about 10x the full-FT RL LR, which puts it in the 1e-5 to 5e-6 range (Tinker starts at 1e-5; PVPO, which is the closest LEGO analogue, used 5e-6 with rank 32 / alpha 64); KL off or tiny (0-0.001); clip-higher at eps_low 0.2 / eps_high 0.28; token-level ("dapo") or constant-normalized ("dr_grpo") loss; no std-normalization, or batch-level std; entropy bonus 0, or at most ~0.001; temperature 1.0. LoRA rank barely matters for RL, so rank 8-32 on all layers including the MLP is fine.

### Cited Findings
**Group size and batch**
- DAPO (large-scale): 512 prompts per rollout step x 16 responses each; mini-batch 512, giving 16 gradient updates per rollout step — [DAPO](https://arxiv.org/html/2503.14476v2)
- PVPO (LLM LEGO assembly, the closest analogue): group size 8, global and PPO mini-batch 32, temperature 1.0, 6 epochs over 2k RL prompts; step count not reported — [PVPO](https://arxiv.org/html/2606.07602v2)
- CAD-Coder: k = 8 candidates per prompt, batch 384, 1 epoch over 150K descriptions — [CAD-Coder](https://arxiv.org/html/2505.19713)
- 1-shot RLVR: 128 prompts per batch x 8 responses = 1024 rollouts per step, mini-batch 128 — [1-shot RLVR](https://arxiv.org/html/2504.20571)
- Tinker RL docs: group_size default 4 (range 2-16). Small groups have higher variance but give every problem some signal; large groups give better advantage estimates at higher cost. num_substeps 1-4. Temperature 1.0 (range 0.7-1.0) — [Tinker RL hyperparams](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- A Tinker example config uses group_size 4 x 16 groups = 64 rollouts per step at LR 1e-5 (third-party guide) — [Sail Research Tinker RL](https://docs.sailresearch.com/tinker-rl)
- TRL GRPOTrainer: num_generations default 8; num_iterations (mu) default 1 — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Fireworks RFT: recommends 4-8 rollouts per prompt (minimum 2); token-based batch default 32k tokens; max tokens default 2048; temperature must be > 0 — [Fireworks RFT params](https://docs.fireworks.ai/fine-tuning/rft-parameters-reference); [Fireworks parameter tuning](https://docs.fireworks.ai/fine-tuning/parameter-tuning)

**Learning rate and LoRA**
- "LoRA Without Regret": in policy-gradient RL, LoRA matches full fine-tuning even at rank 1. The optimal LoRA LR is consistently ~10x the full-FT LR. With alpha/r scaling (alpha = 32) the optimal LR is roughly rank-independent. Apply LoRA to all layers, especially the MLP: attention-only underperformed MLP-only in RL. Constant LR with no warmup or cooldown. LoRA has a wider band of near-optimal LRs than full FT — [Thinking Machines, LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Full-FT GRPO LRs are ~1e-6: DAPO uses a constant 1e-6 with a 20-step linear warmup — [DAPO](https://arxiv.org/html/2503.14476v2); the TRL default is 1e-6 — [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Tinker: LR range 1e-6 to 1e-4; start at 1e-5 — [Tinker RL hyperparams](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- PVPO LoRA GRPO: LR 5e-6, LoRA rank 32 / alpha 64, BF16, 8x RTX 4090 — [PVPO](https://arxiv.org/html/2606.07602v2)
- Fireworks RFT defaults: LoRA rank 8 (suggested 8-32 or 8-128 depending on doc version), LR 1e-4, 1 epoch. The docs disagree slightly between versions — [Fireworks RFT params](https://docs.fireworks.ai/fine-tuning/rft-parameters-reference)

**KL**
- DAPO removes the KL term entirely — [DAPO](https://arxiv.org/html/2503.14476v2). TRL defaults to beta = 0.0 and cites Open-Reasoner-Zero, DAPO and Dr. GRPO showing KL isn't essential; DeepSeek-R1 used 0.001 — [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- PVPO, CAD-Coder and 1-shot RLVR all used beta = 0.001 — [PVPO](https://arxiv.org/html/2606.07602v2); [CAD-Coder](https://arxiv.org/html/2505.19713); [1-shot RLVR](https://arxiv.org/html/2504.20571)
- Tinker defaults to 0.0 but suggests 0.05 as a starting point (range 0.01-0.1). Values of 0.1 or more may slow reward gains — [Tinker](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)

**Clipping, loss aggregation, normalization**
- DAPO clip-higher: eps_low = 0.2, eps_high = 0.28. It counters entropy collapse, where with naive GRPO "the entropy of the policy decreases quickly" and group samples become near-identical — [DAPO](https://arxiv.org/html/2503.14476v2)
- DAPO ablation on AIME24, Qwen2.5-32B: naive GRPO 30 -> +overlong filtering 36 -> +clip-higher 38 -> +soft overlong punishment 41 -> +token-level loss 42 -> +dynamic sampling 50 — [DAPO](https://arxiv.org/html/2503.14476v2)
- Token-level loss rationale: sample-level averaging under-weights tokens in long responses and fails to penalize gibberish or repetition in long outputs — [DAPO](https://arxiv.org/html/2503.14476v2)
- Dr. GRPO: GRPO's per-response 1/|o| length normalization and division by std(r) create length bias (long wrong answers) and question-difficulty bias. The fix is to divide by a constant (max completion length) and drop std scaling. TRL supports loss_type "grpo", "dapo" (the default), "dr_grpo", "sapo", "cispo"; scale_rewards "group" (default), "batch" or False. Lite PPO ("Tricks or Traps?", 2508.08221) found group mean + batch std the most robust — [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer); [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- Tinker mean-centers advantages within the group with no std division, citing Dr. GRPO — [Tinker](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)

**Entropy bonus**
- 1-shot RLVR: an entropy coefficient of -0.001 (i.e. an entropy bonus) raised MATH500 from 70.8 to 74.8 and AIME24 from 15.0 to 17.5. -0.003 was slightly worse (73.6) and the authors warn a larger coefficient may destabilize training. Without entropy, performance plateaued around step 150, when train accuracy saturated. Rollout temperature 0.6; raising it to 1.0 added ~0.8% — [1-shot RLVR](https://arxiv.org/html/2504.20571)
- PVPO used entropy coefficient 0 at temperature 1.0 — [PVPO](https://arxiv.org/html/2606.07602v2)

**Max output length**
- DAPO: 16,384 expected max + 4,096 soft-punishment cache (20,480 hard cap). Truncated samples are masked from the loss ("overlong filtering", +6 AIME points) — [DAPO](https://arxiv.org/html/2503.14476v2)
- PVPO (LEGO): max prompt 300, max response 3,796, sequence 4,096 — [PVPO](https://arxiv.org/html/2606.07602v2)
- TRL exposes mask_truncated_completions (default False) — [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)

### Inferences
- Suggested starting config for SAMgo (LoRA, a few hundred steps): group size 8; 16-32 prompts per step (128-256 rollouts); LR 1e-5 for LoRA rank 16-32 (alpha = 2x rank), all linear layers, constant schedule with a short (~10-step) warmup; beta 0 (or 0.001 if drift is a concern); eps 0.2 / 0.28; loss_type dapo or dr_grpo; scale_rewards False or "batch"; entropy coefficient 0 at temperature 1.0 (add 0.001 only if entropy collapses); one gradient update per rollout batch (mu = 1) to stay on-policy; max response = ~1.2x the longest legitimate SFT edit list. Mask truncated outputs rather than scoring them 0.
- 200-300 steps x ~200 rollouts = 40-60k rollouts. PVPO's 2k prompts x 8 x 6 epochs is about 96k rollouts, so a $100 run is roughly half of PVPO's RL volume.
- If you use the Fireworks managed defaults (LR 1e-4, rank 8), note that 1e-4 is at the top of Tinker's suggested range. Expect to drop it toward 1e-5 if rewards oscillate or format breaks.

### Gaps
- No source gave the GRPO step count for PVPO or CAD-Coder, so "a few hundred steps is enough" is not directly benchmarked for this task class.
- I found no 2025-26 ablation of LoRA rank vs RL quality on structured-output tasks specifically. The rank-1 result is on math.

## 2. Pitfalls: reward hacking, zero-variance groups, length bias, format collapse, entropy collapse

### Takeaway
The biggest risks for this task are (a) optimizing physical validity alone (PhysHack), (b) optimizing geometric similarity alone (format and validity break), (c) zero-variance groups wasting compute, and (d) entropy collapse. The mitigations are a coupled validity+fidelity reward, filtering constant-reward groups (DAPO dynamic sampling / Tinker remove_constant_reward_groups), clip-higher, and Dr. GRPO-style normalization.

### Cited Findings
- PhysHack: assemblies satisfy physical-validity constraints but are geometrically misaligned or semantically inconsistent. The paper frames it as data-induced (from BrickGPT's SFT data) and amplified by RL: physics-only reward (lambda = 0) gave Validity@4 1.00 but Qwen-VL 0.52 and voxel 0.29. Voxel-only (lambda = 1) collapsed validity to 0.79 and voxel to 0.08. The coupled reward (lambda = 0.5) gave validity 0.93, Qwen-VL 0.77, DINOv3 0.80, voxel 0.35. Full-data SFT Qwen already reached 0.93 validity but only 0.68 MiMo / 0.39 Gemma semantic scores — [PVPO](https://arxiv.org/html/2606.07602v2). Note: the abstract calls PhysHack "data-induced"; the physics-only RL ablation shows RL amplifies it — [PVPO abstract](https://arxiv.org/abs/2606.07602)
- CAD-Coder: a Chamfer-distance-only reward led to invalid code after ~200 steps, halting RL. Adding a format reward fixed it — [CAD-Coder](https://arxiv.org/html/2505.19713)
- Zero-variance groups: DAPO dynamic sampling over-samples and discards prompts whose group accuracy is all 0 or all 1 until the batch is full; it was the largest single gain (+8 AIME) — [DAPO](https://arxiv.org/html/2503.14476v2). Tinker: identical rewards in a group produce zero advantage and no gradient; remove_constant_reward_groups filters them (default False) — [Tinker](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- Length bias: GRPO's normalization inflates length, especially of incorrect outputs — [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- Entropy collapse: DAPO observed rapid entropy decline under naive GRPO and fixed it with clip-higher — [DAPO](https://arxiv.org/html/2503.14476v2); 1-shot RLVR plateaued without an entropy bonus — [1-shot RLVR](https://arxiv.org/html/2504.20571)
- Multi-objective hacking: GRPO's advantage is biased toward the higher-variance reward term, so the policy can optimize one objective and neglect the others. MO-GRPO normalizes each reward by its variance — [MO-GRPO](https://arxiv.org/abs/2509.22047v1)
- Aggregated neural or process rewards are hackable: the model score rises while verifier scores stay flat — [VeriGate](https://arxiv.org/pdf/2605.30451); see also [Aggregation-induced reward hacking](https://arxiv.org/pdf/2609.00213)

### Inferences
- For SAMgo: never train on pass/fail checks alone, or the model will learn tiny, trivially stable edits (the PhysHack analogue). Never train on similarity alone, or JSON validity and physics will degrade (the CAD-Coder failure).
- Log per component (JSON-parse rate, each check's pass rate, similarity, edit-count / length, entropy, fraction of zero-variance groups) every step. Stop or adjust when the parse rate drops or the edit count shrinks while reward rises.
- With pass/fail checks plus a continuous similarity term, groups are rarely exactly constant, so zero-variance filtering matters mostly for prompts where everything fails to parse.

### Gaps
- No source quantified PhysHack-style hacking specifically for edit-list (diff) outputs rather than full builds.

## 3. Reward design for multi-component verifiers (gating vs weighted sum, partial credit, dense vs sparse)

### Takeaway
Successful 3D/CAD RL papers use a hard gate on executability or validity (an invalid output scores 0 on fidelity) plus a dense, continuous fidelity term. Within valid outputs they use either a 50/50 linear blend of validity-fraction and fidelity (PVPO) or a piecewise-linear fidelity reward with a small format term (CAD-Coder). Partial credit (fraction of bricks valid, normalized distance) beats binary rewards.

### Cited Findings
- PVPO: R_phys = fraction of bricks passing constraints (partial credit); R_vox = 1 - min(D_vox / d_max, 1), a symmetric voxel Chamfer distance; R = (1 - lambda) R_phys + lambda R_vox with lambda = 0.5; no explicit gate — [PVPO](https://arxiv.org/html/2606.07602v2)
- CAD-Coder: piecewise-linear geometric reward. CD < 1e-5 scores 1.0; CD > 0.5 or failed execution scores 0; CD = 0.5 scores a 0.01 floor, rising linearly in between. Plus a binary format reward (regex for think block + code block). R = lambda_geo R_geo + lambda_fmt R_fmt; execution failure gates geometry to 0 — [CAD-Coder](https://arxiv.org/html/2505.19713)
- ReCode Consistency-Gated GRPO: the process signal counts only when execution succeeds. A linear sum of terms is described as especially exposed to hacking — [ReCode](https://arxiv.org/pdf/2508.05170)
- MO-GRPO: normalize each reward term by its variance so no term dominates the advantage — [MO-GRPO](https://arxiv.org/abs/2509.22047v1)
- TRL supports multiple reward functions combined as a weighted sum via reward_weights — [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Composite verifiable rewards with penalty terms reduced format violations and answer-hacking on 3B models without hurting accuracy — [Composite rewards](https://arxiv.org/pdf/2509.15557)

### Inferences
- Recommended SAMgo reward:
  - R = 0 if the JSON fails to parse or apply (or a small negative such as -0.5 to separate it from "valid but poor").
  - Otherwise R = 0.5 * (fraction of checks passed, or fraction of bricks valid) + 0.5 * similarity, with similarity in [0, 1] using a CAD-Coder-style piecewise floor.
  - Optionally add a +0.2 bonus only when all hard checks pass AND similarity exceeds a threshold, to reward full success.
- Because similarity is continuous, std-normalizing per group can amplify tiny similarity differences into large advantages. Prefer scale_rewards=False or "batch".

### Gaps
- PR-CAD and STEP-LLM were not reviewed (tool-call budget); their reward formulations remain unchecked.

## 4. Curriculum when the initial pass rate is low

### Takeaway
RL needs a non-zero pass rate per prompt: problems the model never solves give no signal. The practical tools are (a) filtering prompts to an intermediate pass rate, (b) reverse curriculum, starting near the solution and moving the start point backward (R3), and (c) an SFT warm start that lifts the pass rate off zero. Very few well-chosen examples can suffice (1-shot RLVR).

### Cited Findings
- 1-shot RLVR: a single example took Qwen2.5-Math-1.5B on MATH500 from 36.0% to 73.6% (matching a 1.2k-example set); two examples reached 74.8%. Examples the model almost never solves (a wrong label, or too difficult) gave no gains. Examples were ranked by historical training-accuracy variance, and the top-16 by variance beat 16 random examples on 7B (42.5 vs 40.2). Train accuracy saturated before step 100, and test accuracy kept improving after saturation ("post-saturation generalization"); overfitting began around step 1400-1800 — [1-shot RLVR](https://arxiv.org/html/2504.20571); [abstract](https://arxiv.org/abs/2504.20571)
- R3 reverse curriculum: starts reasoning near the end of a correct demonstration and slides the start point backward, giving easy exploration with outcome-only reward; +4.1 points average over RL across 8 tasks on Llama2-7B — [R3, ICML 2024](https://proceedings.mlr.press/v235/xi24a.html); [arXiv](https://arxiv.org/abs/2402.05808)
- DAPO dynamic sampling keeps only prompts with 0 < accuracy < 1 — [DAPO](https://arxiv.org/html/2503.14476v2)
- CAD-Coder: GRPO without SFT was much worse (mean CD 76.20 vs 6.54 with SFT+GRPO) — [CAD-Coder](https://arxiv.org/html/2505.19713)

### Inferences
- For SAMgo, reverse curriculum maps naturally to "remove k parts from a reference build, ask the model to restore them": start at k = 1-2 and grow k as the pass rate per bucket rises above ~70%. Keep the active pool where group mean reward / full-pass rate is roughly 10-90% (the DAPO filter, relaxed).
- Pre-measure the pass rate per prompt with the SFT model (8 samples) before RL and drop prompts at 0/8 or 8/8 on full pass.

### Gaps
- No found paper reported a specific "keep pass rate in 10-90%" window as a tuned number; that window is common practice, not sourced. The R3 paper's specific schedule details were not retrieved.

## 5. SFT warm-up size for structured tasks; SFT+RL vs either alone

### Takeaway
Thousands of high-quality SFT examples (8k-11k), not hundreds of thousands, followed by RL on a smaller prompt set (2k), is the pattern in the two closest papers. In both, SFT+RL clearly beats SFT-only and RL-only. Quality filtering of SFT data matters more than quantity.

### Cited Findings
- PVPO: 11k SFT examples (5% of BrickGPT's 213k, selected by model-based data selection), 6 epochs per Table 6 (the text says 66, which conflicts), then RL on 2k prompts deduplicated to one description per object. A PVPO DeepSeek-R1-Distill-Llama-8B beat Kimi-K3 on semantic, structural and physical metrics — [PVPO](https://arxiv.org/html/2606.07602v2)
- CAD-Coder: 8k high-quality SFT samples (CD < 1e-4), 3 epochs, LR 1e-5, batch 64, full-param, then 1.5k CoT samples for 2 epochs, then GRPO. An 8k high-quality set beat 70k medium-quality (mean CD 6.54 vs 9.89). Ablation: SFT-only 74.55 mean CD / 5.33% invalid; GRPO without SFT 76.20 / 5.33%; full pipeline 6.54 / 1.45% — [CAD-Coder](https://arxiv.org/html/2505.19713)

### Inferences
- For SAMgo, a short SFT of ~2-10k filtered edit examples for 2-3 epochs should be enough to make JSON validity near 100% and lift the pass rate above zero. RL then needs only ~1-2k distinct prompts.
- Per the user's own pilot memory, SFT on self-samples was NO-GO twice, so SFT data should come from reference (ground-truth) edits, not model samples.

### Gaps
- No source isolates the minimum SFT size for edit-list outputs specifically.

## 6. Multi-turn / feedback-conditioned RL (verifier messages in context)

### Takeaway
Training with verifier feedback in context works and beats compute-matched single-turn GRPO, mainly when feedback is also used at inference. MURPHY (2-turn GRPO with execution feedback) is the clearest recipe. Some groups (Goedel-Prover-V2) found multi-turn RL not yet worth the complexity and used single-pass self-correction training instead.

### Cited Findings
- MURPHY: turn 1 samples G1 outputs; unsolved ones are re-prompted with the prompt, the previous output and the executor feedback (failing tests, errors), sampling G2 children. 72 rollouts per query, 1,000 KodCode training prompts. Rewards are propagated backward: MaRS sets each node to the max of its own and its best descendant's reward, and matched or beat the discounted-mean variant. Advantages are GRPO-normalized per turn. Qwen3-1.7B HumanEval 80.07 base -> 82.11 GRPO -> 86.58 MURPHY (with feedback at inference); MBPP 53.93 -> 57.20 -> 62.00. Pruning (InterP, mu + sigma ranking) cuts updates with little loss. Gains are mostly in multi-iteration settings; single-turn performance is competitive with GRPO — [MURPHY](https://arxiv.org/html/2511.07833v2)
- MulFeRL: when all K rollouts fail, the group advantage collapses; verbal feedback in a multi-turn loop supplies signal for failed samples — [MulFeRL](https://arxiv.org/pdf/2601.22900)
- Goedel-Prover-V2: multi-turn RL "still require[s] further validation on effectiveness and efficiency"; they used a simpler setup — [Goedel-Prover-V2](https://arxiv.org/pdf/2508.03613)

### Inferences
- This aligns with the user's pilot finding that verifier-feedback retries beat best-of-k (from-empty 0.72 vs 0.56). A cheap MURPHY-style variant for SAMgo:
  - For prompts where turn-1 samples fail, build turn-2 prompts with the verifier error messages.
  - Use them as additional GRPO groups, with the turn-1 reward set to the max of its own and its best child's.
  - This also rescues all-fail groups (the MulFeRL motivation).
- Under the $100 budget, single-turn GRPO first, then 2-turn only if budget remains, is the lower-risk order.

### Gaps
- MURPHY's LR, LoRA usage and step count (Appendix G.2) were not retrieved.
- No found work applies feedback-conditioned RL to 3D/CAD generation.
