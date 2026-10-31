# RL budget for SAMgo

Planning estimate checked October 31, 2026. No paid training was launched or authorized by this estimate. This prices the existing Muse Glimmer 30B path; switching models is a separate experiment.

## Updated constraint: $100 in Prime Intellect RL credits

The user has $100 of Prime Intellect RL credits. Treat this as the available experiment budget; the Fireworks estimates below describe a different provider and cannot be converted directly into Prime steps or hours.

Prime's current documentation says the shared, per-token hosted LoRA service is retired. Its replacement Hosted Training uses dedicated GPUs per run, full-parameter training and GPU-hour billing. Self-managed `prime-rl` still supports LoRA. Older Verifiers tutorials and the May Lab announcement describe the retired service; do not use their model list or configuration as a current launch contract. [Retirement notice](https://docs.primeintellect.ai/hosted-training/shared-lora-retired)

Training credit grants contribute to the spendable wallet and are consumed before the paid balance. On-demand access must be enabled. Rates and model availability are account/cluster dependent and should be inspected with `prime train models --output json`. GPU charges start when pods are ready; storage is separate. The wallet can include an overdraft allowance, so a $100 credit grant is not itself a $100 enforced spend cap. [Billing](https://docs.primeintellect.ai/hosted-training/billing), [model discovery](https://docs.primeintellect.ai/hosted-training/models)

Proposed allocation, pending account eligibility and live rates:

| Purpose | Credit allocation |
|---|---:|
| Baseline evaluation and short environment smoke test | $10 |
| One focused repair-training experiment | $65 |
| Held-out evaluation against its pre-RL checkpoint | $15 |
| Storage, stop latency and contingency | $10 |
| **Total** | **$100** |

For the training portion, maximum planned wall-clock time is `$65 / sum(allocated GPU hourly rates)`, before any separately metered services. Include both trainer and rollout-inference GPUs. This is an allocation, not a rate quote or a claim about achievable optimizer steps. Measure throughput and actual charges during the smoke test before selecting the main step count.

First verify whether the current Glimmer base and training state are usable on Prime. A Fireworks sampler checkpoint identifier is not automatically a portable training checkpoint. If Glimmer is unavailable, a small available Meta/Llama model is a candidate for the hackathon, with a new baseline and curriculum; its results must not be reported as continued training of r3. Preserve checkpoint storage explicitly, since current Hosted Training disables checkpoint saving by default without a volume and checkpoint configuration. [Getting started](https://docs.primeintellect.ai/hosted-training/getting-started)

Keep this run narrow: beam-length/placement repairs and response to verifier feedback on held-out translated layouts. Do not spend the pilot trying to teach every object family simultaneously. The repo's certificate and existing mechanism checks can support a Prime environment adapter, but that adapter and its reward tests have not yet been implemented. The formal static certificate alone does not cover wheel or hinge tasks.

At this update, the local workspace has no Prime CLI, Prime configuration directory or Prime API credential variables. No account balance, grant expiration, access entitlement or live training rate has been inspected, and no resources have been provisioned. The offline calculator below remains explicitly Fireworks-only.

## Current prices and workload

Fireworks lists Glimmer Serverless Training API rates of **$1.96 input, $0.39 cached input, $4.88 generation and $5.86 training per million tokens**. Its advertised Glimmer SFT promotion runs through November 30 and does not say RL is free. [Provider pricing](https://fireworks.ai/pricing)

The provider meters sampler input, generated rollouts and tokens passed to `forward_backward` separately. It supports LoRA RL with custom rewards. The training-token estimate here includes the **entire prompt and completion**, even where only the completion contributes loss. Actual padding and additional passes must be measured on a smoke run. [Serverless training documentation](https://docs.fireworks.ai/fine-tuning/training-api/serverless)

Default planning workload:

- 16 prompts per update, 8 samples per prompt: 128 rollouts per step.
- 3,000 prompt tokens, 550 generated tokens per sample, including any hidden reasoning.
- One forward/backward update per accepted batch, no separate reference-model pass.
- 1,000 evaluation completions total across all compared checkpoints/seeds.
- A 30% reserve applied to training plus evaluation.

The lower scenario assumes one uncached and seven cached prompts in each sample group. The upper scenario assumes all sampler input is uncached. **These are planning scenarios, not guaranteed lower/upper bounds.** Extra sampling to replace zero-variance/rejected groups, multiple updates, padding and longer reasoning can exceed them.

## Compute budget

Calculated from those assumptions:

| Experiment | Training rollouts | Training/sampling before evaluation/reserve | With evaluation and 30% reserve |
|---|---:|---:|---:|
| 10-step smoke test | 1,280 | $32–38 | **$53–60** |
| One 50-step targeted run | 6,400 | $162–188 | **$221–255** |
| One 100-step run | 12,800 | $323–376 | **$431–500** |
| Three independent 50-step runs | 19,200 | $485–564 | **$641–744** |

I would allocate **$100 initially**, stop after the smoke test to inspect real meters and reward variation, then allow **$300 for one targeted 50-step pilot** if the setup works. A **$750–1,000 compute envelope** gives room for three short runs, evaluation and more uncertainty. These amounts do not promise improved transfer or cover a general-purpose assembly learner.

The most useful first comparison is SFT versus SFT+RL on frozen held-out object templates. Before buying more RL steps, generate verified repair demonstrations and confirm that the sampler sometimes solves the training cases. If every sampled answer receives the same reward, GRPO has no within-group advantage signal. The existing narrow static certificate cannot be used as a success reward for unsupported wheel/hinge tasks; those tasks need their explicit checks, and must retain target/function gates.

## Prompt length changes the estimate materially

For 50 steps, one seed, the same 1,000 evaluation samples and reserve:

| Mean prompt/output length | Estimated total |
|---|---:|
| 1,000 / 550 tokens | $109–120 |
| 3,000 / 550 tokens | $221–255 |
| 8,000 / 1,000 tokens | $545–637 |

The recent mechanism probes included large part lists and token-limited outputs. Their visible JSON length is not a safe proxy for billed generation. Short local geometry views and concise valid edit lists could reduce cost, but must preserve the information needed to solve the task. Multi-turn RL must count each trajectory's actual contexts and generated tokens; this table models a single completion per rollout.

## Reproducible calculator

`samgo/rl_budget.py` is entirely offline; it has no credentials, network client or job-launch capability. Prices are date-stamped constants, not dynamically refreshed rates.

```bash
python -m samgo.rl_budget --steps 50 --seeds 1
python -m samgo.rl_budget --steps 50 --seeds 3
python -m samgo.rl_budget --steps 50 --prompt-tokens 1000
python -m samgo.rl_budget --steps 50 --rollout-multiplier 2 \
  --train-sequence-tokens 4096 --eval-samples 2000
```

For `B` prompts, `G` samples, prompt length `P`, output length `C`, train length `T`, rollout multiplier `R`, and updates per batch `U`, the per-step dollar estimate is:

```text
sampling = B * G * R * C * 4.88 / 1e6
training = B * G * T * U * 5.86 / 1e6
cached-input scenario = B * R * P * (1.96 + (G - 1) * 0.39) / 1e6
all-cold input scenario = B * G * R * P * 1.96 / 1e6
```

`T` defaults to `P+C`; override it for padded training sequences. The rollout multiplier charges additional sampling while keeping the accepted training batch fixed. Evaluation is priced all-cold and `--eval-samples` is a total, not per seed. A reserve is not an enforced spend cap; any eventual training loop needs pre-request budget checks and tracking of already submitted work.

These estimates exclude engineering labor, dataset generation, local verifier CPU, physical parts/testing, taxes and any separate SFT/inference services. There is no measured throughput for this project's RL loop yet, so an hours-to-completion estimate would be speculative. Confirm checkpoint-resume compatibility and account billing on the initial bounded smoke test before committing to longer runs.
