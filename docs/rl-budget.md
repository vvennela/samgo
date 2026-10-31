# RL budget for SAMgo

Planning estimate checked October 31, 2026. No paid training was launched or authorized by this estimate. This prices the existing Muse Glimmer 30B path; switching models is a separate experiment.

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
