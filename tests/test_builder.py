"""S4: the algorithmic builder produces verified builds from procedural targets."""

import numpy as np
import pytest
from scipy import ndimage

from samgo.builder import build
from samgo.parts import parse_piece
from samgo.targets import FAMILIES, sample


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_targets_are_one_grounded_solid(family):
    for seed in range(5):
        _, target = sample(seed, family)
        _, n = ndimage.label(target > 0)
        assert n == 1 and (target[:, :, 0] > 0).any()


@pytest.mark.parametrize("family", ["box", "pyramid", "stairs", "table", "chair", "mushroom", "cup"])
def test_builder_passes_regular_families(family):
    for seed in range(5):
        _, target = sample(seed, family)
        result = build(target, seed=seed)
        assert result.report.passed, (family, seed, result.report.issues)


def test_builds_use_only_palette_parts_in_ground_up_order():
    _, target = sample(3, "chair")
    pieces = build(target).pieces
    for raw in pieces:
        parse_piece(raw)
    assert [p["z"] for p in pieces] == sorted(p["z"] for p in pieces)


def test_staggered_seams_tie_a_wall_together():
    # Two bricks wide, two courses tall: stacking 2x2s straight up would make two towers.
    target = np.full((4, 2, 6), 4, dtype=np.int32)
    result = build(target)
    assert result.report.passed and result.report.metrics["components"] == 1


def test_build_is_deterministic():
    _, target = sample(11)
    assert build(target, seed=1).pieces == build(target, seed=1).pieces


def test_overall_pass_rate_meets_s4_bar():
    passed = sum(build(sample(k)[1], seed=k).report.passed for k in range(100))
    assert passed >= 80
