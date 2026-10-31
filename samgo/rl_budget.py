"""Offline Glimmer GRPO planning estimate; never creates a job or spends money.

Prices checked 2026-10-31: https://fireworks.ai/pricing
Meter semantics: https://docs.fireworks.ai/fine-tuning/training-api/serverless
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass

RATES = dict(prefill=1.96, cached_prefill=0.39, sample=4.88, train=5.86)


@dataclass(frozen=True)
class Workload:
    steps: int = 50
    seeds: int = 1
    prompts_per_step: int = 16
    samples_per_prompt: int = 8
    prompt_tokens: int = 3000
    completion_tokens: int = 550
    updates_per_batch: int = 1
    rollout_multiplier: float = 1.0  # rejected groups / additional attempts still cost tokens
    train_sequence_tokens: int | None = None  # allow a padded billing length
    eval_samples: int = 1000  # total across checkpoints/seeds, all-cold estimate
    reserve: float = 0.30

    def __post_init__(self):
        for key in (
            "steps",
            "seeds",
            "prompts_per_step",
            "samples_per_prompt",
            "prompt_tokens",
            "completion_tokens",
            "updates_per_batch",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.eval_samples) is not int or self.eval_samples < 0:
            raise ValueError("eval_samples must be a nonnegative integer")
        if not math.isfinite(self.rollout_multiplier) or self.rollout_multiplier < 1:
            raise ValueError("rollout_multiplier must be finite and at least 1")
        if not math.isfinite(self.reserve) or self.reserve < 0:
            raise ValueError("reserve must be finite and nonnegative")
        if self.train_sequence_tokens is not None and (
            type(self.train_sequence_tokens) is not int
            or self.train_sequence_tokens < self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("train_sequence_tokens cannot be shorter than the prompt plus completion")


def estimate(w=Workload()):
    batch = w.prompts_per_step * w.samples_per_prompt
    train_length = w.train_sequence_tokens or w.prompt_tokens + w.completion_tokens
    sample_cost = batch * w.rollout_multiplier * w.completion_tokens * RATES["sample"] / 1e6
    train_cost = batch * train_length * w.updates_per_batch * RATES["train"] / 1e6
    prefill = w.prompts_per_step * w.rollout_multiplier * w.prompt_tokens / 1e6
    cached = prefill * (RATES["prefill"] + (w.samples_per_prompt - 1) * RATES["cached_prefill"])
    cold = prefill * w.samples_per_prompt * RATES["prefill"]
    eval_cost = (
        w.eval_samples * (w.prompt_tokens * RATES["prefill"] + w.completion_tokens * RATES["sample"]) / 1e6
    )
    scenarios = {}
    for name, prefill_cost in (("one_cold_rest_cached", cached), ("all_cold", cold)):
        step = prefill_cost + sample_cost + train_cost
        training_run = step * w.steps * w.seeds
        subtotal = training_run + eval_cost
        scenarios[name] = dict(
            per_step=round(step, 6),
            step_components=dict(
                prefill=round(prefill_cost, 6), sample=round(sample_cost, 6), train=round(train_cost, 6)
            ),
            runs=round(training_run, 2),
            evaluation=round(eval_cost, 2),
            reserve=round(subtotal * w.reserve, 2),
            total=round(subtotal * (1 + w.reserve), 2),
        )
    return dict(
        price_checked="2026-10-31",
        provider="Fireworks Serverless Training API",
        model="Muse Glimmer 30B",
        source="https://fireworks.ai/pricing",
        rates_per_million_usd=RATES,
        workload=asdict(w),
        planned_training_rollouts=batch * w.steps * w.seeds,
        scenarios=scenarios,
        limitations=[
            "Planning scenarios, not a quote, account balance, spend cap or guarantee of learning.",
            "Assumes full prompt+completion billing for one training pass unless overridden.",
            "Cache behavior, padding, extra logprob passes and rejected groups can increase cost.",
            "Completion tokens must include hidden reasoning, not just visible JSON.",
            "Excludes engineering labor, local verifier CPU, data preparation, taxes and physical validation.",
            "SFT promotion does not make RL free; verify current rates before launch.",
        ],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    defaults = Workload()
    for key in (
        "steps",
        "seeds",
        "prompts_per_step",
        "samples_per_prompt",
        "prompt_tokens",
        "completion_tokens",
        "updates_per_batch",
        "train_sequence_tokens",
        "eval_samples",
    ):
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(defaults, key))
    ap.add_argument("--rollout-multiplier", type=float, default=1)
    ap.add_argument("--reserve", type=float, default=0.30)
    print(json.dumps(estimate(Workload(**vars(ap.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
