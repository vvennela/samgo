import random

from pilot.grpo import advantages, draw


def test_advantages_are_mean_centered_and_skip_flat_groups():
    assert advantages([1, 1, 1, 1]) is None and advantages([0, 0]) is None
    a = advantages([1, 0, 0, 1])
    assert a == [0.5, -0.5, -0.5, 0.5] and abs(sum(a)) < 1e-9


def test_draw_never_reuses_a_prompt():
    rows = [{"id": f"{lv}-{i}", "level": lv} for lv in ("5", "10") for i in range(5)]
    used, rng = set(), random.Random(0)
    first, second = draw(rows, ["5", "10"], 2, rng, used), draw(rows, ["5", "10"], 2, rng, used)
    ids = [r["id"] for r in first + second]
    assert len(ids) == len(set(ids)) == 8
    assert len(draw(rows, ["5"], 9, rng, used)) == 1  # only one level-5 task left
