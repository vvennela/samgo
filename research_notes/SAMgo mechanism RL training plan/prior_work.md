# Prior work: RL / LLM generation of LEGO assemblies and functional mechanisms

## STABLE (arXiv 2605.06947): base model, data, SFT, GRPO, rewards, compute, results, failure modes

### Takeaway
STABLE is the closest recipe to SAMgo. It does full-parameter SFT on Qwen3.5-2B, then TRL GRPO with group size 8 and lr 5e-6, using an additive verifier reward (collision + IoU + interlock + connectivity, total in [-10,10]). This takes collision-free rate from 0.00 after SFT to 0.99, without rollback at inference. RL from the base model fails completely, so SFT has to come first.

### Cited Findings
- Base model is Qwen3.5-2B, full-parameter fine-tuning for both SFT and GRPO. Qwen3.5-122B-A10B is evaluated only as a baseline — [arXiv HTML](https://arxiv.org/html/2605.06947)
- Dataset: PointCloud2Brick, built from StableText2Brick. It has 47,389 physically stable structures across 21 ShapeNet categories, split 42,604 train / 4,785 test — [arXiv HTML](https://arxiv.org/html/2605.06947)
- World: 20x20x20 grid. Bricks are 1 unit tall, with 8 brick types giving 14 oriented variants — [arXiv HTML](https://arxiv.org/html/2605.06947)
- Tokenization: the input point cloud is written as occupied-voxel tokens "(x,y,z)". Output bricks are "h×w (x,y,z)", e.g. "1x4 (5,6,0)" (BrickGPT-style text) — [arXiv HTML](https://arxiv.org/html/2605.06947)
- SFT: 3 epochs, lr 2e-4 cosine, 100 warmup steps, per-device batch 2, grad accum 4, 4 processes (DDP), bf16, max sequence length 8192 — [arXiv HTML](https://arxiv.org/html/2605.06947)
- GRPO setup:
  - TRL GRPOTrainer, starting from the SFT checkpoint.
  - 8 generations per prompt, max completion 8192 tokens, temperature 0.7.
  - lr 5e-6 cosine, 100 warmup steps, 1 epoch, per-device batch 2, grad accum 4, max grad norm 1.0.
  - vLLM colocate (GPU memory utilization 0.4), DeepSpeed ZeRO-2.
  - KL coefficient and total optimizer steps: not reported.
  - [arXiv HTML](https://arxiv.org/html/2605.06947)
- Reward: R = R_col + R_shape + R_inter + R_conn, range [-10, 10] — [arXiv HTML](https://arxiv.org/html/2605.06947)
  - R_col = max(-10, -2·N_col), where N_col is the number of voxels occupied by more than one brick.
  - R_shape = 5·IoU. It is not gated by collision-freeness.
  - R_inter = 3·S_inter, where S_inter is the fraction of non-ground bricks that bridge at least 2 supporting bricks in the layer below. It applies only if the build is collision-free and in bounds.
  - R_conn = 2·(1 − |disconnected voxels|/|occupied voxels|), with the same gating as R_inter.
  - Physical stability is not a training reward. It is an evaluation metric only, computed with a Gurobi-based simulator.
- Compute: training on 4 H100/H200 GPUs, evaluation on an NVIDIA L40S. Training hours and GPU-hours: not reported — [arXiv HTML](https://arxiv.org/html/2605.06947)
- Main results (Table 1), as collision-free / IoU / mean stability / min stability / connectivity / interlock / mean bricks / seconds — [arXiv HTML](https://arxiv.org/html/2605.06947):

  | Method | Coll.-free | IoU | Mean stab. | Min stab. | Conn. | Interlock | Mean bricks | Time (s) |
  |---|---|---|---|---|---|---|---|---|
  | BrickGPT | 0.10 | 0.039 | 0.214 | 0.031 | 0.00 | 0.023 | 110.9 | 32.97 |
  | BrickGPT+rollback&resample | 1.00 | 0.349 | 0.983 | 0.781 | 0.54 | 0.148 | 168.6 | 895.41 |
  | LegoACE | 0.05 | 0.080 | 0.041 | 0.007 | 0.12 | 0.103 | 79.3 | n/a |
  | Gemini-3.1-Pro | 0.02 | 0.276 | 0.913 | 0.643 | 0.00 | 0.119 | 64.32 | n/a |
  | Qwen3.5-2B base, Qwen3.5-122B, Llama-4 | 0.00 | 0.000 | – | – | – | – | 0 | – |
  | STABLE-ZERO (RL from base) | 0.00 | 0.000 | – | – | – | – | – | – |
  | STABLE-SFT | 0.00 | 0.813 | 0.947 | 0.228 | 0.05 | 0.012 | 661.5 | 183.63 |
  | STABLE (SFT+GRPO) | 0.99 | 0.907 | 1.000 | 1.000 | 0.97 | 0.722 | 216.5 | 57.64 |
- Ablations (Table 2), as collision-free / mean colliding voxels / IoU / connectivity ratio / interlock / mean bricks — [arXiv HTML](https://arxiv.org/html/2605.06947):
  - Without collision term: 0.72 / 14.51 / 0.897 / 0.91 / 0.734 / 247.1
  - Without connectivity term: 0.93 / 9.50 / 0.919 / 0.73 / 0.709 / 230.0
  - Without structural terms (shape only): 0.96 / 12.46 / 0.978 / 0.46 / 0.069 / 387.0, and in-bounds rate 0.76
- Failure modes — [arXiv HTML](https://arxiv.org/html/2605.06947):
  - SFT alone gives high IoU but redundant, overlapping, weakly connected bricks.
  - RL from the base model fails because the base model rarely produces parseable brick sequences.
  - A shape-only reward is gamed by filling with many 1x1 bricks (387 bricks on average).
- Limitations stated by the authors: fixed 20^3 world and a predefined brick library; the rewards do not model friction, deformation or external forces — [arXiv HTML](https://arxiv.org/html/2605.06947)

### Inferences
- For SAMgo, the STABLE settings are a reasonable default starting point: G=8, lr 5e-6, temperature 0.7, 1 epoch, full fine-tune on a ~2B model, SFT lr 2e-4 for 3 epochs.
- Two reward-design lessons carry over:
  - Additive, dense, partially gated terms worked. Structural terms gated on collision-free/in-bounds prevented reward hacking.
  - SAMgo's mechanism verifier should likewise include a "don't wreck the body" term, so the policy can't spam parts.
- "SFT on teacher data, then RL on verifier reward" is validated here. Note that SFT used ground-truth stable data, not self-samples, which is consistent with SAMgo's NO-GO on SFT from self-samples.
- STABLE's speed claim needs care. 57.64 s vs 895.41 s is about a 94% cut, not "orders of magnitude", and STABLE is slower than plain BrickGPT (32.97 s).

### Gaps
- KL coefficient, total GRPO steps, number of RL prompts used, and wall-clock/GPU-hours are not reported.
- No editing task: STABLE generates from scratch, conditioned on a point cloud, and does not model mechanisms.

## BesiegeField / "Agentic Design of Compositional Machines" (arXiv 2510.14980): environment, agents, RL setup, cold start, results

### Takeaway
LLMs build functional machines in a Besiege physics sim from a 27-block library that includes wheels, hinges and springs, using a construction-tree DSL. Simulator-state feedback plus iterative refining works better than one-shot generation. Cold-start SFT followed by GRPO (LoRA r64, lr 5e-6, KL 0.001, 400 steps, about 48 h) on Qwen2.5-14B mostly raises the best-of-N score, not validity or the mean, and reduces diversity.

### Cited Findings
- Environment — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - Built on the game Besiege (Unity), using a filtered 27-block library: structural blocks, joints, powered and unpowered wheels, hinges, springs, grabbers, containers, boulders, ballast.
  - Machines settle under gravity, then powered parts activate on a fixed open-loop schedule.
  - 7 task families: Movement, Throwing, Delivery, Jumping, Lifting, Flying, Gearing.
- Representation: a construction-tree DSL in which each new part names its parent, the parent face, and its orientation. It outperformed a global-coordinate format; for example, Gemini 2.5 Pro scored 8.49 mean in tree format vs 4.96 in coordinates, and o3, Claude Opus 4 and Llama 4 Scout produced 0/8 valid machines in coordinate format — [arXiv HTML](https://arxiv.org/html/2510.14980)
- Agent workflows — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - Single-agent: CoT, then the full program.
  - Iterative editing: a designer proposes, an environment-query module summarizes the sim result, and a refiner proposes several candidate edits; the best branch is kept.
  - Hierarchical: a meta-designer splits the machine into modules, which are built with beam search.
  - Useful feedback was sim state (distance, trajectory, orientation drift, broken parts). The authors say a scalar reward alone is less useful.
- Zero-shot construction-tree results, as file valid / 3D valid / final valid / mean / max — [arXiv HTML](https://arxiv.org/html/2510.14980):

  | Model | File valid | 3D valid | Final valid | Mean | Max |
  |---|---|---|---|---|---|
  | Gemini 2.5 Pro | 8/8 | 5/8 | 5/8 | 8.49 | 9.14 |
  | Kimi K2 | 6/8 | 6/6 | 6/8 | 7.18 | 12.02 |
  | Claude Opus 4 | 8/8 | 4/8 | 4/8 | 4.17 | 4.36 |
  | o3 | 3/8 | 3/3 | 3/8 | 0 | 0 |
  | DeepSeek-V3 | 7/8 | 7/7 | 7/8 | 0 | 0 |
- Cold-start SFT — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - Qwen2.5-14B-Instruct trained on 9,982 valid machine-CoT pairs, generated by Gemini 2.5 Pro from 100 prompts and filtered for validity and function.
  - QOFT (OFT block size 64), 8-bit AdamW, lr 1e-6, 3% warmup. Epochs: not reported.
  - Compute: "8 A100 GPUs for 72 hours".
- RL — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - verl GRPO, lr 5e-6, grad clip 0.5, clip ratio 0.2, KL 0.001 to the initial policy, no entropy bonus.
  - LoRA rank 64 on all linear layers.
  - Temperature 1.0, top-p 0.95, max input 3440 tokens, max output 1168 tokens.
  - "400 update steps", about 48 h. RL GPU count and group size: not reported.
  - A Pass@k-style advantage variant (rollouts equal to k; Pass@64 was best on Throwing) found high-reward designs better than standard GRPO.
- Rewards — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - Invalid machines get 0.
  - Movement: distance traveled by the root block along the target direction.
  - Throwing: valid only if the boulder's max height is above 3 m; score is max height × max horizontal distance.
- Table 18 (50 generations each), as valid / mean / max — [arXiv HTML](https://arxiv.org/html/2510.14980):

  | Setting | Throwing | Movement |
  |---|---|---|
  | Base | 11/50, 0.06, 2.41 | 46/50, 4.97, 19.10 |
  | Cold-start | 9/50, 0.11, 5.54 | 40/50, 4.67, 20.23 |
  | RL from base | 12/50, 0.13, 5.92 | 41/50, 3.72, 24.08 |
  | Cold-start + RL | 11/50, 0.14, 7.14 | 42/50, 5.05, 45.72 |
- Failure modes — [arXiv HTML](https://arxiv.org/html/2510.14980):
  - Wrong part orientations or parent attachments, missing supports.
  - CoT says one mechanism but the program builds another.
  - Parts look plausible but cannot transfer motion.
  - LLMs stay in local editing loops.
  - RL mostly refines existing templates. Entropy and diversity fall.
  - Removing KL or cold start gives "more failed rollouts and slower improvement" (Fig. 38).
  - Rewards are sparse and heavy-tailed, so means have wide CIs; e.g. Movement cold-start+RL mean 95% CI is [2.89, 7.21].

### Inferences
- Of all the work reviewed, this is the closest to "LLM builds working mechanisms graded by a sim/verifier". The lesson is that functional rewards are sparse and heavy-tailed. SAMgo's binary or near-binary mechanism checks (does the axle spin, does the hinge rotate) are cleaner than travel distance and should give denser signal.
- Parent-relative attachment (construction tree) beat absolute coordinates by a wide margin. This is relevant if SAMgo's edits specify mechanism parts relative to existing bricks.
- Their cold-start data is 9,982 teacher (Gemini) samples, not self-samples, and cold start alone did not help (validity dropped 46→40/50 on Movement). Gains appeared only after RL.

### Gaps
- RL group size, batch size, and GPU count were not found in the text read.
- Per-step curves (Figs. 32 and 38) are not available as numbers.

## PVPO (arXiv 2606.07602): anything new from related citations

### Takeaway
Nothing new was gathered. PVPO was already read and I did not pursue its citations further within the call budget.

### Cited Findings
- None.

### Inferences
- None.

### Gaps
- The PVPO citation graph was not followed.

## BrickNet, LegoACE, LEGO-Maker, BrickSim: what they do and any mechanism modeling

### Takeaway
BrickNet is the only one of the four that represents hinge, axle and ball connections, as typed connectors in a graph program, and it is SFT-only with no motion evaluation. LegoACE (with DPO) and LEGO-Maker (with rollback) are autoregressive brick generators with no mechanisms. BrickSim is a real-time snap-fit physics simulator with no Technic or joint modeling mentioned.

### Cited Findings
- **BrickNet** (CVPR 2026, Kulits & Schmid) — [arXiv HTML](https://arxiv.org/html/2604.22984); [abs](https://arxiv.org/abs/2604.22984):
  - Models: Qwen3 instruct, at 0.6B / 1.7B / 4B / 8B / 14B.
  - Data: PT set of 320,808 samples, 9,743 part variants, 40,549,969 placed bricks, with sequences capped at 100 pieces. SFT set of 67,185 samples captioned by Gemini 2.5. 512 held-out eval samples. All drawn from over 100,000 human LDraw models.
  - Representation: a spanning tree over the connectivity graph. Each part attaches through a typed connector from 5 families: stud, hinge (with a flip parameter), axle (with a slide), ball (2 extra rotational DoF), and fixed.
  - Training: supervised next-token prediction only, no RL. Hyperparameters and compute: not reported.
  - Results: graph models stay connectivity-valid for about 94–97 steps vs 32–50 for pose-based models, but only about 16–22 steps once collision checks are added.
  - Text-conditioned invalid-placement rate is 0.231–0.260 for BrickNet vs 0.063 for BrickGPT. The BrickGPT row looked odd in the extracted table, so treat it with caution.
  - Moving connections are represented but not evaluated as motion.
- **LegoACE** (SIGGRAPH Asia 2025, VAST) — [project page](https://xh38.github.io/LegoACE/); [HF](https://huggingface.co/VAST-AI/LegoACE):
  - Decoder-only autoregressive transformer conditioned on text or multi-view normal maps.
  - LegoVerse dataset: over 55,000 models, 9,314 brick types.
  - Each step outputs position, rotation and type. The released vocabulary covers 28 common brick types and 20 rotation classes.
  - Uses DPO fine-tuning.
  - Connectivity is implicit. The model card warns outputs "may produce structurally unstable or non-buildable arrangements".
  - In STABLE's benchmark it scored 0.05 collision-free and 0.080 IoU ([STABLE](https://arxiv.org/html/2605.06947)).
  - Model size: not reported on the page.
- **LEGO-Maker** (TOG 44(6), Art. 201, Dec 2025, Chi-Wing Fu lab, CUHK) — [ACM DL](https://dl.acm.org/doi/10.1145/3763285):
  - Image-conditioned autoregressive model over 100+ brick types, generating hundreds of bricks.
  - Uses a custom tokenization, multi-token prediction, and a rollback mechanism for collision-free output.
  - No mechanism modeling is mentioned in the abstract.
- **BrickSim** (arXiv 2603.16853) — [abs](https://arxiv.org/abs/2603.16853):
  - Real-time snap-fit physics in Isaac Sim, using a force model solved as a convex QP. Average solve time is about 5 ms.
  - 100% static-stability accuracy on 150 real assemblies; reproduces breakage in drop tests.
  - Aimed at robotic assembly. No Technic, joints or LLM use is mentioned in the abstract.

### Inferences
- BrickNet's connector taxonomy (stud / hinge+flip / axle+slide / ball+2DoF / fixed) is a direct template for SAMgo's mechanism vocabulary and verifier checks.
- None of this prior work verifies that a mechanism actually moves, so a SAMgo verifier that does would be a real contribution relative to these.

### Gaps
- LegoACE model size and DPO details are missing because the PDF was too large to fetch.
- LEGO-Maker's full text (data size, metrics) was not read; it is behind ACM DL.
- BrickSim's full text was not read.

## LLMs designing functional LEGO/Technic mechanisms, or repairing/editing 3D/CAD with verifier feedback

### Takeaway
I found no work where an LLM designs functional LEGO/Technic mechanisms; BesiegeField is the nearest analogue. In CAD, CAD-Coder gives a concrete GRPO recipe: Qwen2.5-7B, 8K SFT pairs, G=8, KL 0.001, about 146 h on 8 A800s, and a Chamfer-threshold reward. Its key warning is that a geometry-only reward led to invalid code after about 200 steps, and adding a format/CoT reward was needed. Iterative-edit CAD agents (IterCAD, CADReasoner) and Articraft (articulated assets via code plus tests) do verifier-feedback loops, but I could only read their abstracts or summaries.

### Cited Findings
- **CAD-Coder** (arXiv 2505.19713) — [arXiv HTML](https://arxiv.org/html/2505.19713):
  - Base: Qwen2.5-7B-Instruct.
  - SFT: 8K high-quality pairs (CD < 1e-4), chosen over a 70K medium-quality set; this gave mean CD 6.54 vs 9.89 with the larger set. 3 epochs, batch 64, lr 1e-5, AdamW, full fine-tuning with ZeRO-2.
  - CoT cold start: 1.5K samples, 2 epochs, batch 384.
  - GRPO: starts from SFT, 1 epoch over all 150K Text2CAD prompts, batch 384, k=8, KL β=0.001. RL lr, clip ε and step count: not reported.
  - Geometric reward: 1.0 if CD < 1e-5; 0 if CD > 0.5 or execution fails; 0.01 at CD = 0.5; linear in between. Format reward is 1 for a `<think>` block plus a code block.
  - Compute: 8× A800 80GB; SFT about 7 h, GRPO about 146 h.
  - Results, as mean CD×10³ / invalid rate: CAD-Coder 6.54 / 1.45%, Text2CAD 29.29 / 3.75%, GPT-4o 133.52 / 93.00%, Claude-3.7-sonnet 186.53 / 47.03%, untuned Qwen2.5-7B 202.35 / 98.83%.
  - Ablations: SFT only gives 74.55 / 5.33%; GRPO without CoT gives 17.34 / 4.95%.
  - Failure: with CD as the only reward, generated code became invalid after about 200 GRPO steps and training halted (App. B). Thin or hollow geometry allowed reward hacking through sparse point sampling.
  - No iterative self-correction loop.
- **IterCAD**: a multimodal agent that generates and edits CadQuery code in a closed loop using compiler, execution and visual feedback. It is trained with GSPO on a Chamfer-distance reward, with prefix masking for multi-turn credit assignment, and ablations credit these for self-correction. This comes from a secondary summary only; the arXiv id and numbers were not obtained (fetch got 429) — [themoonlight summary](https://www.themoonlight.io/en/review/itercad-an-iterative-multimodal-agent-for-visually-grounded-cad-generation-and-editing)
- **CADReasoner** (arXiv 2603.29847): iteratively edits a runnable CadQuery program, feeding the rendered mesh back at each step. Evaluated on DeepCAD, Fusion 360 and MCB, with SOTA claimed. Training details are not in the abstract — [abs](https://arxiv.org/abs/2603.29847)
- **Memory-augmented RL CAD agent** (arXiv 2605.19748): combines a geometric-kernel toolchain with case/skill memory and RL-trained retrieval — [papers.cool](https://papers.cool/arxiv/2605.19748)
- **Articraft** (arXiv 2605.15187, May 2026): the LLM writes a program against an SDK that defines parts and joints and writes tests. The harness validates the result and returns structured feedback. Produced the Articraft-10K dataset (over 10K articulated assets, 245 categories). SFT/RL not described in the abstract — [abs](https://arxiv.org/abs/2605.15187)
- Articulate-Anything is described in related work as LLM code generation for articulated objects, refined in a loop with simulation feedback — [Kinematify related work](https://arxiv.org/html/2511.01294v2)
- A targeted search for LLMs generating LEGO Technic mechanisms returned only general Technic references, no research papers — [web search, e.g. Wikipedia: Lego Technic](https://en.wikipedia.org/wiki/Lego_Technic)

### Inferences
- Across STABLE, BesiegeField and CAD-Coder, three hyperparameters recur: G=8, KL around 0.001, policy lr 5e-6 (STABLE and BesiegeField).
- Across the same papers, SFT data before RL ranges from about 8K (CAD-Coder) through about 10K (BesiegeField) to about 42K (STABLE) examples.
- SAMgo's task (edit an existing build to add a verified mechanism) is not directly covered. The edit/repair-with-verifier pattern exists in CAD (IterCAD, CADReasoner) and articulated assets (Articraft), but not for LEGO.
- CAD-Coder's collapse with a geometry-only reward suggests keeping an explicit parse/format-valid term in SAMgo's reward, and watching the invalid-output rate during GRPO.

### Gaps
- IterCAD's arXiv id, hyperparameters and turn counts could not be verified; only a secondary summary was read.
- CADReasoner's and Articraft's training details and quantitative results were not read (abstracts only).
- No paper was found on LLM design of LEGO Technic gears, axles or ball joints with motion verification.
