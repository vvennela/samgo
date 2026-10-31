import pytest

from samgo.rl_budget import Workload, estimate


def test_full_sequence_training_is_charged_even_when_only_output_has_loss():
    result = estimate(
        Workload(
            steps=1,
            seeds=1,
            prompts_per_step=1,
            samples_per_prompt=1,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            eval_samples=0,
            reserve=0,
        )
    )
    # 1M input + 1M output + a 2M-token forward/backward pass.
    assert result["scenarios"]["all_cold"]["total"] == pytest.approx(1.96 + 4.88 + 2 * 5.86)
    assert result["scenarios"]["all_cold"] == result["scenarios"]["one_cold_rest_cached"]


def test_rejected_rollouts_padding_evaluation_and_reserve_increase_estimate():
    base = estimate(Workload())["scenarios"]["all_cold"]
    costly = estimate(
        Workload(rollout_multiplier=2, train_sequence_tokens=4096, eval_samples=2000, reserve=0.50)
    )["scenarios"]["all_cold"]
    assert costly["step_components"]["sample"] == 2 * base["step_components"]["sample"]
    assert costly["step_components"]["train"] > base["step_components"]["train"]
    assert costly["total"] > base["total"]
    with pytest.raises(ValueError):
        Workload(train_sequence_tokens=550)
